"""POST /simulate, /simulate/ligand, /simulate/batch — MD job submission."""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from ..state import AppState, get_state
from ..auth import validate_api_key
from ..models import MDSimulationRequest, LigandPreparationRequest, BatchMDRequest
from ..pdb_handling import fetch_pdb_from_rcsb
from ..queue import estimate_runtime_minutes, get_gpu_queue_info
from ..redis_jobs import update_job_status

logger = logging.getLogger(__name__)

router = APIRouter()


from ..simulation import run_simulation_pipeline


async def _simulate_with_fetch(
    state: AppState,
    job_id: str,
    compound_id: str,
    pdb_id: str,
    pdb_content: str,
    ligand_smiles: str,
    simulation_ns: float,
    temperature: float,
    pressure: float,
):
    """Fetch PDB (if needed) and dispatch to the simulation pipeline.

    Classification and protein-only cleaning both happen inside
    `run_simulation_pipeline` — this function's only job is the RCSB
    fetch step. Previously called `clean_pdb_protein_only()` inline here
    which silently stripped metals/cofactors; that's now done after
    classification approves the structure.
    """
    try:
        if pdb_id and not pdb_content:
            await update_job_status(state, job_id, "running", {
                "percentage": 2,
                "message": f"Fetching PDB {pdb_id} from RCSB",
                "step": "fetch_pdb",
            })
            pdb_content = await fetch_pdb_from_rcsb(pdb_id)

        await run_simulation_pipeline(
            state,
            job_id=job_id,
            compound_id=compound_id,
            pdb_content=pdb_content,
            ligand_smiles=ligand_smiles,
            simulation_ns=simulation_ns,
            temperature=temperature,
            pressure=pressure,
            pdb_id=pdb_id,
        )
    except Exception as e:
        logger.error(f"Job {job_id} failed during fetch/prep: {e}")
        await update_job_status(state, job_id, "failed", {
            "percentage": 0,
            "message": str(e),
            "step": "error",
        })


@router.post("/simulate", dependencies=[Depends(validate_api_key)])
async def submit_simulation(request: MDSimulationRequest, state: AppState = Depends(get_state)):
    """Submit MD simulation (runs asynchronously in background).

    Returns job_id immediately — PDB fetch and all prep happens in the
    background task so the MCP gateway never times out waiting.
    """
    compound_id = request.compound_id or (
        f"{request.pdb_id}-ligand" if request.pdb_id else f"mol-{uuid.uuid4().hex[:8]}"
    )
    job_id = f"gro_{compound_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if not request.pdb_id and not request.pdb_content and not request.ligand_smiles:
        raise HTTPException(status_code=400, detail="Either pdb_id/pdb_content or ligand_smiles must be provided")

    has_protein = bool(request.pdb_id or request.pdb_content)
    has_ligand = bool(request.ligand_smiles)
    gpu_info = get_gpu_queue_info(state)
    est_minutes = estimate_runtime_minutes(
        request.simulation_ns,
        has_protein=has_protein,
        has_ligand=has_ligand,
        concurrent_jobs=gpu_info["active_jobs"],
    )

    # Initialize job in Redis FIRST — before any slow I/O
    # Store estimate + submitted_at so status endpoint can compute remaining time
    await update_job_status(state, job_id, "queued", {
        "percentage": 0,
        "message": "Simulation queued — fetching protein structure",
        "step": "queued",
    })
    if state.redis:
        try:
            key = f"novomcp:job:{job_id}"
            await state.redis.hset(key, mapping={
                "submitted_at": datetime.utcnow().isoformat(),
                "estimated_runtime_minutes": str(est_minutes),
            })
        except Exception:
            pass

    # Launch background task — PDB fetch + prep + simulation all happen here
    asyncio.create_task(_simulate_with_fetch(
        state,
        job_id=job_id,
        compound_id=compound_id,
        pdb_id=request.pdb_id,
        pdb_content=request.pdb_content,
        ligand_smiles=request.ligand_smiles,
        simulation_ns=request.simulation_ns,
        temperature=request.temperature,
        pressure=request.pressure,
    ))

    return {
        "job_id": job_id,
        "status": "queued",
        "compound_id": compound_id,
        "simulation_ns": request.simulation_ns,
        "estimated_runtime_minutes": est_minutes,
        "gpu_queue": gpu_info,
        "poll_url": f"/gromacs-md/status/{job_id}",
    }


@router.post("/simulate/ligand", dependencies=[Depends(validate_api_key)])
async def simulate_ligand(request: LigandPreparationRequest, state: AppState = Depends(get_state)):
    """Submit ligand-only MD simulation."""

    compound_id = f"ligand-{uuid.uuid4().hex[:8]}"
    job_id = f"gro_{compound_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    gpu_info = get_gpu_queue_info(state)
    est_minutes = estimate_runtime_minutes(
        request.duration_ns,
        has_protein=False,
        has_ligand=True,
        concurrent_jobs=gpu_info["active_jobs"],
    )

    await update_job_status(state, job_id, "queued", {
        "percentage": 0, "message": "Ligand simulation queued", "step": "queued",
    })
    if state.redis:
        try:
            key = f"novomcp:job:{job_id}"
            await state.redis.hset(key, mapping={
                "submitted_at": datetime.utcnow().isoformat(),
                "estimated_runtime_minutes": str(est_minutes),
            })
        except Exception:
            pass

    asyncio.create_task(run_simulation_pipeline(
        state,
        job_id=job_id,
        compound_id=compound_id,
        pdb_content=None,
        ligand_smiles=request.smiles,
        simulation_ns=request.duration_ns,
        temperature=request.temperature,
        pressure=1.0,
    ))

    return {
        "job_id": job_id,
        "status": "queued",
        "compound_id": compound_id,
        "smiles": request.smiles,
        "duration_ns": request.duration_ns,
        "estimated_runtime_minutes": est_minutes,
        "gpu_queue": gpu_info,
        "poll_url": f"/gromacs-md/status/{job_id}",
    }


@router.post("/simulate/batch", dependencies=[Depends(validate_api_key)])
async def submit_batch_simulations(request: BatchMDRequest, state: AppState = Depends(get_state)):
    """Submit multiple MD simulations."""
    results = []

    for compound in request.compounds:
        compound_id = compound.get("compound_id", f"mol-{uuid.uuid4().hex[:8]}")
        job_id = f"gro_{compound_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        try:
            await update_job_status(state, job_id, "queued", {
                "percentage": 0, "message": "Simulation queued", "step": "queued",
            })

            asyncio.create_task(run_simulation_pipeline(
                state,
                job_id=job_id,
                compound_id=compound_id,
                pdb_content=compound.get("pdb_content"),
                ligand_smiles=compound.get("smiles") or compound.get("ligand_smiles"),
                simulation_ns=request.simulation_ns,
                temperature=compound.get("temperature", 300.0),
                pressure=compound.get("pressure", 1.0),
                pdb_id=compound.get("pdb_id"),
            ))

            results.append({
                "compound_id": compound_id,
                "job_id": job_id,
                "status": "queued",
            })
        except Exception as e:
            results.append({"compound_id": compound_id, "error": str(e)})

    return {
        "submitted": len([r for r in results if "job_id" in r]),
        "failed": len([r for r in results if "error" in r]),
        "results": results,
    }
