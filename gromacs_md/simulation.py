"""Core simulation pipelines — soluble and membrane."""

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from intake import classify_structure

from .config import NTOMP
from .state import AppState
from .redis_jobs import update_job_status, complete_job, fail_job, refuse_job
from .s3_storage import upload_results_to_blob
from .subprocess_utils import run_subprocess
from .pdb_handling import (
    clean_pdb_protein_only,
    repair_missing_atoms,
    smiles_to_pdb,
)
from .topology import (
    process_protein_with_gromacs,
    generate_ligand_topology,
    combine_protein_ligand,
    create_ligand_only_topology,
    write_em_mdp,
    write_nvt_mdp,
    write_npt_mdp,
    write_md_mdp,
    _fill_mdp_template as fill_mdp_template,
)
from .analysis import analyze_trajectory

logger = logging.getLogger(__name__)


# ─── Step 5: resume-after-restart helpers ────────────────────────────────────
# These power the batch-job resume path. The legacy FastAPI
# background-task callers ignore them (defaults are None throughout), so
# behavior is identical for the existing `/simulate` route.

SCRATCH_DIR = os.getenv("SCRATCH_DIR", tempfile.gettempdir())


def _stage_complete(workspace: Path, stage: str) -> bool:
    """Resume gate: True if the stage has a DONE marker in this workspace.

    Written last by _finalize_stage so a partial copy never looks complete.
    """
    return (workspace / f"stage_{stage}" / "DONE").exists()


def _finalize_stage(
    workspace: Path,
    stage: str,
    file_patterns: Optional[List[str]] = None,
) -> Path:
    """Copy stage outputs into workspace/stage_{stage}/ + write DONE marker.

    file_patterns is a list of glob patterns relative to workspace
    (e.g. [f'{compound_id}_em.*']). If None, copies every file/dir in
    workspace root EXCEPT existing stage_* subdirs — used for system_prep
    which produces many heterogeneous artifacts (topol.top, *.itp,
    ligand.acpype/, multiple .pdb / .gro files).

    The DONE marker is written LAST so a partial copy doesn't get
    treated as a complete stage on resume.
    """
    stage_dir = workspace / f"stage_{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    def _copy(src: Path, dest: Path) -> None:
        if dest.exists():
            return
        if src.is_dir():
            shutil.copytree(src, dest)
        else:
            shutil.copy2(src, dest)

    if file_patterns is None:
        for entry in workspace.iterdir():
            if entry.name.startswith("stage_"):
                continue
            _copy(entry, stage_dir / entry.name)
    else:
        for pattern in file_patterns:
            for src in workspace.glob(pattern):
                if src.name.startswith("stage_"):
                    continue
                _copy(src, stage_dir / src.name)

    (stage_dir / "DONE").write_text(datetime.now(timezone.utc).isoformat())
    return stage_dir


def _restore_workspace_from_stages(workspace: Path) -> None:
    """Flatten downloaded stage_*/ subdirs back to workspace root.

    Called once after a resume_workdir is set, before any GROMACS stage
    runs. The blob download placed files in workspace/stage_*/; the
    existing pipeline expects them flat at workspace root (GROMACS
    commands reference files like `{compound_id}_em.gro` not
    `stage_em/{compound_id}_em.gro`).
    """
    for stage_dir in sorted(workspace.glob("stage_*")):
        if not stage_dir.is_dir():
            continue
        for entry in stage_dir.iterdir():
            if entry.name == "DONE":
                continue
            dest = workspace / entry.name
            if dest.exists():
                continue
            if entry.is_dir():
                shutil.copytree(entry, dest)
            else:
                shutil.copy2(entry, dest)


def _save_system_state(workspace: Path, gro_file: Path, top_file: Path) -> None:
    """Persist the system_prep output filenames so resume can find them.

    The pipeline derives gro_file / top_file paths inside system_prep
    based on the input mode (protein-only, ligand-only, complex). On
    resume past system_prep, we read them back here rather than
    re-deriving — avoids having to thread the input-mode discriminator
    through resume.
    """
    (workspace / "_md_state.json").write_text(json.dumps({
        "gro_file": gro_file.name,
        "top_file": top_file.name if hasattr(top_file, "name") else str(top_file),
    }))


def _load_system_state(workspace: Path):
    """Read the system_prep output filenames after resume. Returns (gro, top) or (None, None)."""
    state_file = workspace / "_md_state.json"
    if not state_file.exists():
        return None, None
    data = json.loads(state_file.read_text())
    return workspace / data["gro_file"], workspace / data["top_file"]


def _check_npt_density_plateau(workspace: Path, compound_id: str) -> Dict:
    """Sync helper for adaptive NPT: extract density from npt.edr and apply
    the analysis._density_plateau detector.

    Runs `gmx energy` non-interactively (input "Density\\n" selects the
    density group) and reads the produced .xvg. Returns a dict with at
    least `converged: bool`, plus the underlying _density_plateau result
    on success. Returns `{converged: False, reason: ...}` on any failure
    so the adaptive loop fails closed (keep extending) rather than
    declaring premature convergence.
    """
    from gromacs_md.analysis import _density_plateau

    xvg = workspace / f"{compound_id}_npt_density_check.xvg"
    try:
        subprocess.run(
            ["gmx", "energy", "-f", f"{compound_id}_npt.edr", "-o", xvg.name],
            input=b"Density\n",
            cwd=workspace,
            capture_output=True,
            timeout=60,
        )
    except Exception as e:
        return {"converged": False, "reason": f"gmx energy failed: {e}"}
    if not xvg.exists():
        return {"converged": False, "reason": "no density xvg written"}
    rows: list = []
    for line in xvg.read_text(errors="replace").splitlines():
        if line.startswith(("#", "@")) or not line.strip():
            continue
        parts = line.split()
        try:
            rows.append([float(parts[0]), float(parts[1])])
        except (ValueError, IndexError):
            continue
    if len(rows) < 8:
        return {"converged": False, "reason": f"too few density samples ({len(rows)})"}
    plateau = _density_plateau(rows)
    return {
        "converged": bool(plateau.get("detected")),
        "plateau": plateau,
        "n_samples": len(rows),
    }


async def _production_cpt_uploader(
    workspace: Path,
    compound_id: str,
    stage_dir: Path,
    callback: Callable[[Path], None],
    interval_s: int = 300,
) -> None:
    """Periodic md.cpt copy + blob upload during production.

    Runs as an asyncio task while gmx mdrun is active. Every interval_s
    seconds, copies {compound_id}_md.cpt → stage_dir/md.cpt and invokes
    callback(cpt_path). Cancelled cleanly when mdrun exits — the finally
    block in the production stage handles awaiting cancellation.

    On retry after preemption, the most recent uploaded md.cpt drives
    `gmx mdrun -cpi md.cpt -append` so the trajectory continues rather
    than restarting from t=0. Worst-case lost work: one interval_s
    window (~5 min wall-clock).
    """
    cpt = workspace / f"{compound_id}_md.cpt"
    dest = stage_dir / "md.cpt"
    while True:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return
        try:
            if cpt.is_file():
                stage_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cpt, dest)
                callback(cpt)
        except Exception as e:
            logger.warning(f"Production cpt upload tick failed: {e}")


async def _run_membrane_pipeline(
    state: AppState,
    job_id: str,
    compound_id: str,
    pdb_content: str,
    simulation_ns: float,
    temperature: float,
):
    """Membrane MD pipeline — CHARMM36m protein + lipids via packmol-memgen.

    Separate from the soluble pipeline because:
    - Bilayer construction replaces box + solvate + ionize
    - Semi-isotropic pressure coupling (not isotropic)
    - Six-stage equilibration (not two-stage)
    - CHARMM36m force field (not AMBER99SB-ILDN)
    - Membrane-specific analysis (area-per-lipid, SCD, density profile)

    Runs inside the GPU semaphore like the soluble pipeline.
    """
    from intake.membrane import build_membrane_system, convert_to_gromacs

    loop = asyncio.get_event_loop()

    async with state.gpu_sem:
        logger.info(f"Job {job_id}: Acquired GPU semaphore (membrane branch)")

        try:
            with tempfile.TemporaryDirectory() as workspace_str:
                workspace = Path(workspace_str)
                ntomp = str(NTOMP)
                mdp_dir = Path("/app/mdp_templates/membrane")

                # 1. Write protein PDB to workspace
                protein_pdb = workspace / f"{compound_id}_protein.pdb"
                protein_pdb.write_text(pdb_content)

                # 2. Build bilayer with packmol-memgen
                await update_job_status(state, job_id, "processing", {
                    "percentage": 8,
                    "message": "Building lipid bilayer (packmol-memgen)",
                    "step": "bilayer_build",
                })
                try:
                    system_pdb, system_prmtop, build_stats = await loop.run_in_executor(
                        None,
                        build_membrane_system,
                        protein_pdb, workspace, compound_id,
                    )
                except Exception as e:
                    logger.error(f"Job {job_id}: bilayer build failed: {e}")
                    await fail_job(state, job_id, f"Bilayer construction failed: {e}")
                    return

                await update_job_status(state, job_id, "processing", {
                    "percentage": 12,
                    "message": f"Bilayer built: {build_stats.get('n_lipids', '?')} lipids",
                    "step": "bilayer_done",
                })

                # 3. Convert AMBER topology to GROMACS format
                await update_job_status(state, job_id, "processing", {
                    "percentage": 15,
                    "message": "Converting topology (parmed AMBER→GROMACS)",
                    "step": "topology_convert",
                })
                try:
                    gro_file, top_file = await loop.run_in_executor(
                        None,
                        convert_to_gromacs,
                        system_prmtop, system_pdb, workspace, compound_id,
                    )
                except Exception as e:
                    logger.error(f"Job {job_id}: topology conversion failed: {e}")
                    await fail_job(state, job_id, f"Topology conversion failed: {e}")
                    return

                # 4. Energy minimization
                await update_job_status(state, job_id, "processing", {
                    "percentage": 18,
                    "message": "Energy minimization",
                    "step": "em",
                })
                em_mdp = mdp_dir / "em.mdp"
                await loop.run_in_executor(
                    None, run_subprocess,
                    ["gmx", "grompp", "-f", str(em_mdp),
                     "-c", str(gro_file), "-p", str(top_file),
                     "-o", f"{compound_id}_em.tpr", "-maxwarn", "5"],
                    workspace,
                )
                await loop.run_in_executor(
                    None, run_subprocess,
                    ["gmx", "mdrun", "-deffnm", f"{compound_id}_em",
                     "-ntmpi", "1", "-ntomp", ntomp, "-nb", "gpu"],
                    workspace,
                )

                # 5. NVT equilibration (restrained, 500 ps)
                await update_job_status(state, job_id, "processing", {
                    "percentage": 25,
                    "message": "NVT equilibration (restrained, 500 ps)",
                    "step": "nvt",
                })
                nvt_mdp = fill_mdp_template(mdp_dir / "nvt_restrained.mdp", temperature=temperature)
                nvt_mdp_path = workspace / "nvt.mdp"
                nvt_mdp_path.write_text(nvt_mdp)
                await loop.run_in_executor(
                    None, run_subprocess,
                    ["gmx", "grompp", "-f", str(nvt_mdp_path),
                     "-c", f"{compound_id}_em.gro", "-r", f"{compound_id}_em.gro",
                     "-p", str(top_file),
                     "-o", f"{compound_id}_nvt.tpr", "-maxwarn", "5"],
                    workspace,
                )
                await loop.run_in_executor(
                    None, run_subprocess,
                    ["gmx", "mdrun", "-deffnm", f"{compound_id}_nvt",
                     "-ntmpi", "1", "-ntomp", ntomp, "-nb", "gpu"],
                    workspace,
                )

                # 6. NPT equilibration stages (3 stages, ~15 ns total)
                npt_stages = [
                    ("npt_eq1.mdp", "NPT eq1 (protein + lipid restrained, 5 ns)", 35),
                    ("npt_eq2.mdp", "NPT eq2 (protein restrained, lipids free, 5 ns)", 50),
                    ("npt_eq3.mdp", "NPT eq3 (backbone only, 5 ns)", 65),
                ]
                prev_gro = f"{compound_id}_nvt.gro"
                for mdp_name, stage_msg, pct in npt_stages:
                    stage_id = mdp_name.replace(".mdp", "")
                    await update_job_status(state, job_id, "processing", {
                        "percentage": pct,
                        "message": stage_msg,
                        "step": stage_id,
                    })
                    stage_mdp = fill_mdp_template(mdp_dir / mdp_name, temperature=temperature)
                    stage_mdp_path = workspace / f"{stage_id}.mdp"
                    stage_mdp_path.write_text(stage_mdp)
                    await loop.run_in_executor(
                        None, run_subprocess,
                        ["gmx", "grompp", "-f", str(stage_mdp_path),
                         "-c", prev_gro, "-r", prev_gro,
                         "-p", str(top_file),
                         "-o", f"{compound_id}_{stage_id}.tpr", "-maxwarn", "5"],
                        workspace,
                    )
                    await loop.run_in_executor(
                        None, run_subprocess,
                        ["gmx", "mdrun", "-deffnm", f"{compound_id}_{stage_id}",
                         "-ntmpi", "1", "-ntomp", ntomp, "-nb", "gpu"],
                        workspace,
                    )
                    prev_gro = f"{compound_id}_{stage_id}.gro"

                # 7. Production MD
                nsteps = int(simulation_ns * 1e6 / 2)  # dt=0.002 ps
                await update_job_status(state, job_id, "processing", {
                    "percentage": 70,
                    "message": f"Production MD ({simulation_ns} ns)",
                    "step": "production",
                })
                prod_mdp = fill_mdp_template(
                    mdp_dir / "production.mdp",
                    temperature=temperature,
                    nsteps=nsteps,
                )
                prod_mdp_path = workspace / "production.mdp"
                prod_mdp_path.write_text(prod_mdp)
                await loop.run_in_executor(
                    None, run_subprocess,
                    ["gmx", "grompp", "-f", str(prod_mdp_path),
                     "-c", prev_gro, "-p", str(top_file),
                     "-o", f"{compound_id}_md.tpr", "-maxwarn", "5"],
                    workspace,
                )
                await loop.run_in_executor(
                    None, run_subprocess,
                    ["gmx", "mdrun", "-deffnm", f"{compound_id}_md",
                     "-ntmpi", "1", "-ntomp", ntomp, "-nb", "gpu"],
                    workspace,
                )

                # 8. Analysis (reuse existing + membrane-specific)
                await update_job_status(state, job_id, "processing", {
                    "percentage": 90,
                    "message": "Analyzing trajectory",
                    "step": "analysis",
                })
                analysis = await loop.run_in_executor(
                    None, analyze_trajectory, workspace, compound_id
                )
                analysis["membrane"] = {
                    "build_stats": build_stats,
                    "equilibration_stages": ["em", "nvt", "npt_eq1", "npt_eq2", "npt_eq3"],
                    "force_field": "charmm36m",
                    "lipid": build_stats.get("lipid", "POPC"),
                    "note": "Validate: area-per-lipid 65-68 A^2, density profile symmetry, SCD order parameters",
                }

                # 9. Upload results
                result = {
                    "compound_id": compound_id,
                    "simulation_ns": simulation_ns,
                    "temperature": temperature,
                    "pipeline": "membrane",
                    "force_field": "charmm36m",
                    "analysis": analysis,
                }

                blob_url = await upload_results_to_blob(state, workspace, job_id, result)
                if blob_url:
                    result["blob_url"] = blob_url

                await complete_job(state, job_id, result)

        except Exception as e:
            logger.exception(f"Membrane pipeline failed for {job_id}: {e}")
            await fail_job(state, job_id, str(e))


async def run_simulation_pipeline(
    state: AppState,
    job_id: str,
    compound_id: str,
    pdb_content: Optional[str],
    ligand_smiles: Optional[str],
    simulation_ns: float,
    temperature: float,
    pressure: float,
    pdb_id: Optional[str] = None,
    # ── Resume + checkpoint kwargs (Step 5; legacy callers pass None) ──
    resume_workdir: Optional[Path] = None,
    completed_stages: Optional[List[str]] = None,
    checkpoint_callback: Optional[Callable[[str, Path], None]] = None,
    production_cpt_callback: Optional[Callable[[Path], None]] = None,
    # Scientific intent — drives use-case-specific quality grading in
    # analyze_trajectory's three-layer quality report. None grades all
    # known intents without highlighting any one. See analysis._KNOWN_INTENTS.
    intent: Optional[str] = None,
    # Adaptive equilibration (Ship 2, opt-in): extend NPT in 100ps chunks
    # until density plateau is detected, up to a 1ns cap. When False
    # (default), NPT runs the legacy fixed 100ps.
    adaptive_equilibration: bool = False,
):
    """Full simulation pipeline running in background with GPU semaphore.

    Intake classification runs *before* the GPU semaphore is acquired,
    so refused systems don't hold up the queue. Only the happy-path
    soluble route proceeds to the GROMACS pipeline; everything else
    (membrane proteins, metalloproteins, heme, Fe-S clusters) produces
    a structured refusal result via `refuse_job()`.

    The classifier is the universal gatekeeper — applies to both
    RCSB-fetched PDBs and direct user uploads. Ligand-only jobs
    (pdb_content=None) skip classification entirely.

    Resume kwargs (Step 5, used by run_md_job.py):
      resume_workdir         pre-populated workspace from blob download
      completed_stages       stage names already done in prior attempts
      checkpoint_callback    fires after each stage's _finalize_stage
                             with (stage_name, stage_dir) so the Job
                             executor can upload to Blob + update manifest
      production_cpt_callback fires every 5 min during production with
                             the md.cpt path; enables sub-stage resume

    Legacy callers (routes/simulate.py) pass None for all resume kwargs,
    yielding the original behavior unchanged.
    """
    loop = asyncio.get_event_loop()
    completed = list(completed_stages or [])

    # Skip intake on resume if system_prep already completed — the classifier
    # already approved this protein on the first attempt. If system_prep is
    # NOT complete, re-run intake so pdb2gmx receives the cleaned/repaired
    # PDB even on the retry container.
    skip_intake = "system_prep" in completed

    # ------------------------------------------------------------------
    # Intake classification gate
    # ------------------------------------------------------------------
    route = "run_soluble"  # default for ligand-only (no pdb_content)
    if pdb_content and not skip_intake:
        await update_job_status(state, job_id, "processing", {
            "percentage": 2,
            "message": "Classifying system (membrane / metals / cofactors)",
            "step": "classify",
        })
        try:
            decision = await classify_structure(
                pdb_content=pdb_content,
                pdb_id=pdb_id,
                redis_client=state.redis,
            )
        except Exception as e:
            # Hard infrastructure failure — treat as a job failure, not
            # a classification refusal. Classifier is supposed to return
            # RoutingDecision for all normal cases, including parse errors.
            logger.error(f"Job {job_id}: classifier raised unexpectedly: {e}")
            await fail_job(state, job_id, f"classifier error: {e}")
            return

        route = decision.route
        if route not in ("run_soluble", "run_membrane"):
            await refuse_job(state, job_id, decision)
            return

        # Classifier approved. Next steps in order:
        #
        # 1. Rebuild missing sidechain atoms via PDBFixer. Crystal
        #    structures often have incomplete residues (disordered
        #    density, truncated loops) that break pdb2gmx. The classifier
        #    gates biology; PDBFixer gates structural completeness.
        # 2. Protein-only normalization (altloc handling, drop waters)
        #    that pdb2gmx expects.
        #
        # Both are safe now because the classifier confirmed there's
        # nothing load-bearing to strip.
        await update_job_status(state, job_id, "processing", {
            "percentage": 3,
            "message": "Repairing missing sidechain atoms (PDBFixer)",
            "step": "repair",
        })
        repaired_pdb, repair_stats = await loop.run_in_executor(
            None, repair_missing_atoms, pdb_content
        )
        if "error" in repair_stats:
            logger.info(f"Job {job_id}: PDBFixer skipped ({repair_stats['error']})")
        else:
            added = repair_stats.get("added_atoms", 0)
            residues = repair_stats.get("residues_with_added_atoms", 0)
            if added > 0:
                logger.info(
                    f"Job {job_id}: PDBFixer rebuilt {added} atoms across "
                    f"{residues} residues"
                )
                await update_job_status(state, job_id, "processing", {
                    "percentage": 4,
                    "message": (
                        f"Repaired {added} missing sidechain atoms "
                        f"on {residues} residues"
                    ),
                    "step": "repair_done",
                })
            pdb_content = repaired_pdb

        pdb_content = clean_pdb_protein_only(pdb_content)

    # ------------------------------------------------------------------
    # Membrane branch — separate pipeline (packmol-memgen + CHARMM36m)
    # ------------------------------------------------------------------
    # Membrane resume is out of scope for Phase 1 (DURABILITY-DESIGN Part 2
    # non-goals). On resume we skip the membrane dispatch since the soluble
    # pipeline is the only one with stage gating; the caller is responsible
    # for not passing resume_workdir for membrane jobs.
    if route == "run_membrane" and pdb_content and not skip_intake:
        await _run_membrane_pipeline(
            state=state,
            job_id=job_id,
            compound_id=compound_id,
            pdb_content=pdb_content,
            simulation_ns=simulation_ns,
            temperature=temperature,
        )
        return

    async with state.gpu_sem:
        logger.info(f"Job {job_id}: Acquired GPU semaphore, starting simulation")

        # Resolve workspace: explicit resume_workdir or fresh mkdtemp.
        # Persistent across the function call (NOT a context manager) so the
        # workspace survives across container retries when resume_workdir is
        # passed back in. Success path cleans up explicitly; failure path
        # leaves it for the next retry (which downloads from Blob, since
        # this container's filesystem is gone by then anyway).
        if resume_workdir is not None:
            workspace = Path(resume_workdir)
            workspace.mkdir(parents=True, exist_ok=True)
            _restore_workspace_from_stages(workspace)
            logger.info(
                f"Job {job_id}: Resuming in workspace {workspace}; "
                f"completed_stages={completed}"
            )
        else:
            workspace_str = tempfile.mkdtemp(
                dir=SCRATCH_DIR if Path(SCRATCH_DIR).is_dir() else None,
                prefix=f"md_{job_id}_",
            )
            workspace = Path(workspace_str)

        clean_workspace_on_success = True
        gro_file: Optional[Path] = None
        top_file: Optional[Path] = None

        try:
            # ============== Stage: system_prep ==============
            if "system_prep" in completed or _stage_complete(workspace, "system_prep"):
                if "system_prep" not in completed:
                    completed.append("system_prep")
                gro_file, top_file = _load_system_state(workspace)
                if (
                    gro_file is None or top_file is None
                    or not gro_file.exists() or not top_file.exists()
                ):
                    raise RuntimeError(
                        "Resume marked system_prep complete but _md_state.json "
                        "or referenced files are missing — checkpoint may be corrupt"
                    )
                logger.info(
                    f"Job {job_id}: system_prep skipped (resume); "
                    f"gro={gro_file.name}, top={top_file.name}"
                )
            else:
                await update_job_status(state, job_id, "processing", {
                    "percentage": 5, "message": "Preparing inputs", "step": "preparation",
                })

                if pdb_content and ligand_smiles:
                    # Protein-ligand complex
                    # Use cleaned protein PDB directly (no split needed when pdb_content is pre-cleaned)
                    protein_pdb = workspace / f"{compound_id}_protein.pdb"
                    protein_pdb.write_text(pdb_content)

                    # Generate ligand 3D structure from SMILES
                    ligand_pdb = workspace / f"{compound_id}_ligand.pdb"
                    await loop.run_in_executor(None, smiles_to_pdb, ligand_smiles, ligand_pdb)

                    await update_job_status(state, job_id, "processing", {
                        "percentage": 8, "message": "Processing protein with GROMACS", "step": "pdb2gmx",
                    })
                    protein_gro, protein_top = await loop.run_in_executor(
                        None, process_protein_with_gromacs, workspace, compound_id, protein_pdb
                    )

                    await update_job_status(state, job_id, "processing", {
                        "percentage": 10, "message": "Generating ligand topology", "step": "ligand_topology",
                    })
                    ligand_gro, ligand_itp = await loop.run_in_executor(
                        None, generate_ligand_topology, workspace, compound_id, ligand_pdb, ligand_smiles
                    )

                    gro_file, top_file = await loop.run_in_executor(
                        None, combine_protein_ligand, workspace, compound_id,
                        protein_gro, ligand_gro, protein_top, ligand_itp,
                    )

                elif ligand_smiles:
                    # Ligand-only
                    ligand_pdb = workspace / f"{compound_id}_ligand.pdb"
                    await loop.run_in_executor(None, smiles_to_pdb, ligand_smiles, ligand_pdb)

                    await update_job_status(state, job_id, "processing", {
                        "percentage": 8, "message": "Generating ligand topology", "step": "ligand_topology",
                    })
                    ligand_gro, ligand_itp = await loop.run_in_executor(
                        None, generate_ligand_topology, workspace, compound_id, ligand_pdb, ligand_smiles
                    )
                    top_file = await loop.run_in_executor(
                        None, create_ligand_only_topology, workspace, compound_id, ligand_itp
                    )
                    gro_file = ligand_gro

                elif pdb_content:
                    # Protein-only
                    pdb_path = workspace / f"{compound_id}.pdb"
                    pdb_path.write_text(pdb_content)

                    await update_job_status(state, job_id, "processing", {
                        "percentage": 8, "message": "Processing protein with GROMACS", "step": "pdb2gmx",
                    })
                    gro_file, top_file = await loop.run_in_executor(
                        None, process_protein_with_gromacs, workspace, compound_id, pdb_path
                    )
                else:
                    raise ValueError("Either pdb_content or ligand_smiles must be provided")

                # --- GROMACS MD pipeline ---
                # Box
                await update_job_status(state, job_id, "processing", {
                    "percentage": 10, "message": "Creating simulation box", "step": "editconf",
                })
                await loop.run_in_executor(
                    None, run_subprocess,
                    ["gmx", "editconf", "-f", str(gro_file), "-o", f"{compound_id}_box.gro",
                     "-c", "-d", "1.0", "-bt", "cubic"],
                    workspace,
                )

                # Solvate
                await update_job_status(state, job_id, "processing", {
                    "percentage": 12, "message": "Solvating system", "step": "solvate",
                })
                await loop.run_in_executor(
                    None, run_subprocess,
                    ["gmx", "solvate", "-cp", f"{compound_id}_box.gro", "-cs", "spc216.gro",
                     "-o", f"{compound_id}_solv.gro", "-p", str(top_file)],
                    workspace,
                )

                # Solvate's output is the input for EM. Record the final
                # gro_file pointer for resume so the EM stage finds it after
                # _load_system_state on a retry.
                gro_file = workspace / f"{compound_id}_solv.gro"
                _save_system_state(workspace, gro_file, top_file)

                _finalize_stage(workspace, "system_prep")  # copy everything
                if checkpoint_callback:
                    try:
                        checkpoint_callback("system_prep", workspace / "stage_system_prep")
                    except Exception as e:
                        logger.warning(f"checkpoint_callback(system_prep) failed: {e}")
                completed.append("system_prep")

            ntomp = str(NTOMP)

            # ============== Stage: em ==============
            if "em" in completed or _stage_complete(workspace, "em"):
                if "em" not in completed:
                    completed.append("em")
                logger.info(f"Job {job_id}: em skipped (resume)")
            else:
                await update_job_status(state, job_id, "processing", {
                    "percentage": 15, "message": "Energy minimization", "step": "em",
                })
                write_em_mdp(workspace, compound_id)
                await loop.run_in_executor(
                    None, run_subprocess,
                    ["gmx", "grompp", "-f", f"{compound_id}_em.mdp",
                     "-c", f"{compound_id}_solv.gro", "-p", str(top_file),
                     "-o", f"{compound_id}_em.tpr", "-maxwarn", "5"],
                    workspace,
                )
                await loop.run_in_executor(
                    None, run_subprocess,
                    ["gmx", "mdrun", "-v", "-deffnm", f"{compound_id}_em",
                     "-nb", "gpu", "-update", "cpu", "-gpu_id", "0", "-ntmpi", "1", "-ntomp", ntomp],
                    workspace,
                )
                _finalize_stage(workspace, "em", [f"{compound_id}_em.*"])
                if checkpoint_callback:
                    try:
                        checkpoint_callback("em", workspace / "stage_em")
                    except Exception as e:
                        logger.warning(f"checkpoint_callback(em) failed: {e}")
                completed.append("em")

                await update_job_status(state, job_id, "processing", {
                    "percentage": 25, "message": "Energy minimization complete", "step": "em_done",
                })

            # ============== Stage: nvt ==============
            if "nvt" in completed or _stage_complete(workspace, "nvt"):
                if "nvt" not in completed:
                    completed.append("nvt")
                logger.info(f"Job {job_id}: nvt skipped (resume)")
            else:
                await update_job_status(state, job_id, "processing", {
                    "percentage": 30, "message": "NVT equilibration", "step": "nvt",
                })
                write_nvt_mdp(workspace, compound_id, temperature)
                await loop.run_in_executor(
                    None, run_subprocess,
                    ["gmx", "grompp", "-f", f"{compound_id}_nvt.mdp",
                     "-c", f"{compound_id}_em.gro", "-r", f"{compound_id}_em.gro",
                     "-p", str(top_file), "-o", f"{compound_id}_nvt.tpr", "-maxwarn", "5"],
                    workspace,
                )
                await loop.run_in_executor(
                    None, run_subprocess,
                    ["gmx", "mdrun", "-v", "-deffnm", f"{compound_id}_nvt",
                     "-nb", "gpu", "-update", "cpu", "-gpu_id", "0", "-ntmpi", "1", "-ntomp", ntomp],
                    workspace,
                )
                _finalize_stage(workspace, "nvt", [f"{compound_id}_nvt.*"])
                if checkpoint_callback:
                    try:
                        checkpoint_callback("nvt", workspace / "stage_nvt")
                    except Exception as e:
                        logger.warning(f"checkpoint_callback(nvt) failed: {e}")
                completed.append("nvt")

                await update_job_status(state, job_id, "processing", {
                    "percentage": 45, "message": "NVT equilibration complete", "step": "nvt_done",
                })

            # ============== Stage: npt ==============
            # Two paths: legacy (fixed 100ps) and adaptive (Ship 2, opt-in).
            # Adaptive runs an initial 50ps block, checks density plateau via
            # gmx energy → analysis._density_plateau, and extends in 100ps
            # chunks until plateau or a 1ns cap. The cap is intentionally
            # generous; in practice TIP3P water plateaus within 200-400ps for
            # most ligand-only and small protein-ligand systems.
            if "npt" in completed or _stage_complete(workspace, "npt"):
                if "npt" not in completed:
                    completed.append("npt")
                logger.info(f"Job {job_id}: npt skipped (resume)")
            else:
                await update_job_status(state, job_id, "processing", {
                    "percentage": 50, "message": "NPT equilibration", "step": "npt",
                })

                npt_adaptive_log = None
                if adaptive_equilibration:
                    initial_nsteps = 50000  # 50 ps initial window
                    extension_ps = 100.0    # extend by 100 ps per iteration
                    max_cumulative_ps = 1000.0  # 1 ns cap

                    write_npt_mdp(workspace, compound_id, temperature, pressure, nsteps=initial_nsteps)
                    await loop.run_in_executor(
                        None, run_subprocess,
                        ["gmx", "grompp", "-f", f"{compound_id}_npt.mdp",
                         "-c", f"{compound_id}_nvt.gro", "-r", f"{compound_id}_nvt.gro",
                         "-t", f"{compound_id}_nvt.cpt",
                         "-p", str(top_file), "-o", f"{compound_id}_npt.tpr", "-maxwarn", "5"],
                        workspace,
                    )

                    cumulative_ps = 0.0
                    iteration = 0
                    npt_adaptive_log = []
                    converged = False
                    while True:
                        iteration += 1
                        mdrun_cmd = [
                            "gmx", "mdrun", "-v", "-deffnm", f"{compound_id}_npt",
                            "-nb", "gpu", "-update", "cpu", "-gpu_id", "0",
                            "-ntmpi", "1", "-ntomp", ntomp,
                        ]
                        if iteration > 1:
                            mdrun_cmd.extend(["-cpi", f"{compound_id}_npt.cpt", "-append"])
                        await loop.run_in_executor(None, run_subprocess, mdrun_cmd, workspace)

                        cumulative_ps += (initial_nsteps * 0.001) if iteration == 1 else extension_ps

                        check = await loop.run_in_executor(
                            None, _check_npt_density_plateau, workspace, compound_id,
                        )
                        converged = bool(check.get("converged"))
                        npt_adaptive_log.append({
                            "iteration": iteration,
                            "cumulative_ps": round(cumulative_ps, 1),
                            "converged": converged,
                            "reason": check.get("reason"),
                        })
                        logger.info(
                            f"Job {job_id}: NPT adaptive iter {iteration}, cumulative={cumulative_ps}ps, "
                            f"converged={converged}"
                        )

                        if converged:
                            break
                        if cumulative_ps >= max_cumulative_ps:
                            logger.warning(
                                f"Job {job_id}: NPT adaptive cap of {max_cumulative_ps}ps reached "
                                f"without density plateau — continuing with current state."
                            )
                            break

                        # Extend the .tpr by extension_ps and loop
                        await loop.run_in_executor(
                            None, run_subprocess,
                            ["gmx", "convert-tpr", "-s", f"{compound_id}_npt.tpr",
                             "-extend", str(extension_ps),
                             "-o", f"{compound_id}_npt.tpr"],
                            workspace,
                        )

                    # Persist adaptive trace alongside the workspace state file
                    try:
                        (workspace / "_npt_adaptive.json").write_text(json.dumps({
                            "enabled": True,
                            "iterations": iteration,
                            "cumulative_ps": cumulative_ps,
                            "converged": converged,
                            "log": npt_adaptive_log,
                        }))
                    except Exception as e:
                        logger.debug(f"Could not persist npt adaptive log: {e}")
                else:
                    # Legacy: single fixed 100 ps NPT, unchanged
                    write_npt_mdp(workspace, compound_id, temperature, pressure)
                    await loop.run_in_executor(
                        None, run_subprocess,
                        ["gmx", "grompp", "-f", f"{compound_id}_npt.mdp",
                         "-c", f"{compound_id}_nvt.gro", "-r", f"{compound_id}_nvt.gro",
                         "-t", f"{compound_id}_nvt.cpt",
                         "-p", str(top_file), "-o", f"{compound_id}_npt.tpr", "-maxwarn", "5"],
                        workspace,
                    )
                    await loop.run_in_executor(
                        None, run_subprocess,
                        ["gmx", "mdrun", "-v", "-deffnm", f"{compound_id}_npt",
                         "-nb", "gpu", "-update", "cpu", "-gpu_id", "0", "-ntmpi", "1", "-ntomp", ntomp],
                        workspace,
                    )

                _finalize_stage(workspace, "npt", [f"{compound_id}_npt.*", "_npt_adaptive.json"])
                if checkpoint_callback:
                    try:
                        checkpoint_callback("npt", workspace / "stage_npt")
                    except Exception as e:
                        logger.warning(f"checkpoint_callback(npt) failed: {e}")
                completed.append("npt")

                await update_job_status(state, job_id, "processing", {
                    "percentage": 65, "message": "NPT equilibration complete", "step": "npt_done",
                })

            # ============== Stage: production ==============
            # The dominant cost (~78% of total runtime). Two MD-specific
            # additions vs FEP's window loop:
            #   1. If a partial md.cpt exists from a prior attempt (downloaded
            #      from Blob), continue with -cpi -append rather than starting
            #      from npt output.
            #   2. Spawn a 5-min uploader task that copies md.cpt to the stage
            #      dir and calls production_cpt_callback. Cancelled cleanly
            #      after gmx mdrun exits.
            if "production" in completed or _stage_complete(workspace, "production"):
                if "production" not in completed:
                    completed.append("production")
                logger.info(f"Job {job_id}: production skipped (resume)")
            else:
                await update_job_status(state, job_id, "processing", {
                    "percentage": 67, "message": f"Production MD ({simulation_ns}ns)", "step": "production",
                })

                production_cpt = workspace / f"{compound_id}_md.cpt"
                production_tpr = workspace / f"{compound_id}_md.tpr"
                is_production_resume = production_cpt.exists() and production_tpr.exists()

                if not is_production_resume:
                    write_md_mdp(workspace, compound_id, simulation_ns, temperature, pressure)
                    await loop.run_in_executor(
                        None, run_subprocess,
                        ["gmx", "grompp", "-f", f"{compound_id}_md.mdp",
                         "-c", f"{compound_id}_npt.gro", "-t", f"{compound_id}_npt.cpt",
                         "-p", str(top_file), "-o", f"{compound_id}_md.tpr", "-maxwarn", "5"],
                        workspace,
                    )

                stage_dir = workspace / "stage_production"
                stage_dir.mkdir(parents=True, exist_ok=True)
                cpt_task = None
                if production_cpt_callback is not None:
                    cpt_task = asyncio.create_task(_production_cpt_uploader(
                        workspace, compound_id, stage_dir, production_cpt_callback,
                    ))

                mdrun_cmd = [
                    "gmx", "mdrun", "-v", "-deffnm", f"{compound_id}_md",
                    "-nb", "gpu", "-update", "cpu", "-gpu_id", "0",
                    "-ntmpi", "1", "-ntomp", ntomp,
                    "-nsteps", str(int(simulation_ns * 500000)),
                ]
                if is_production_resume:
                    mdrun_cmd.extend(["-cpi", f"{compound_id}_md.cpt", "-append"])
                    logger.info(
                        f"Job {job_id}: production resume via -cpi -append from {production_cpt}"
                    )

                try:
                    await loop.run_in_executor(None, run_subprocess, mdrun_cmd, workspace)
                finally:
                    if cpt_task is not None:
                        cpt_task.cancel()
                        try:
                            await cpt_task
                        except (asyncio.CancelledError, Exception):
                            pass

                _finalize_stage(workspace, "production", [f"{compound_id}_md.*"])
                if checkpoint_callback:
                    try:
                        checkpoint_callback("production", workspace / "stage_production")
                    except Exception as e:
                        logger.warning(f"checkpoint_callback(production) failed: {e}")
                completed.append("production")

            # ============== Stage: analysis ==============
            await update_job_status(state, job_id, "processing", {
                "percentage": 90, "message": "Production MD complete, analyzing trajectory",
                "step": "analysis",
            })

            # Analyze
            # Pass ligand + route hints so MM-GBSA + ligand dynamics (Theo P1)
            # can run when appropriate. Membrane systems skip MM-GBSA.
            has_ligand = bool(ligand_smiles)
            has_protein = bool(pdb_content)
            top_file_name = str(top_file.name) if hasattr(top_file, "name") else str(top_file) if top_file else None
            # MM-GBSA requires a protein-ligand complex — skip for ligand-only
            # runs to avoid AttributeError on missing protein residues.
            if has_ligand and not has_protein:
                top_file_name = None  # signals analyze_trajectory to skip MM-GBSA
            # has_protein is the literal "protein in the system" flag — set
            # before top_file_name was conditionally zeroed for the MM-GBSA
            # gate above. analyze_trajectory needs both signals: top_file_name
            # gates MM-GBSA execution, has_protein gates adequacy grading.
            _has_protein_for_adequacy = bool(pdb_content)
            analysis = await loop.run_in_executor(
                None,
                lambda: analyze_trajectory(
                    workspace,
                    compound_id,
                    has_ligand=has_ligand,
                    top_file=top_file_name,
                    route=route,
                    simulation_ns=simulation_ns,
                    intent=intent,
                    has_protein=_has_protein_for_adequacy,
                ),
            )

            result = {
                "job_id": job_id,
                "compound_id": compound_id,
                "simulation_completed": True,
                "simulation_ns": simulation_ns,
                "temperature": temperature,
                "pressure": pressure,
                "analysis": analysis,
                "output_files": [
                    f"{compound_id}_md.xtc",
                    f"{compound_id}_md.gro",
                    f"{compound_id}_md.edr",
                    f"{compound_id}_md.log",
                    f"{compound_id}_md.cpt",
                    f"{compound_id}_md.tpr",
                ],
                "resumed_from_stages": list(completed_stages) if completed_stages else None,
            }

            # Upload to object storage
            await update_job_status(state, job_id, "processing", {
                "percentage": 95, "message": "Uploading results", "step": "upload",
            })
            output_location = await upload_results_to_blob(state, workspace, job_id, result)
            if output_location:
                result["output_location"] = output_location

            # Mark complete
            await complete_job(state, job_id, result)

        except subprocess.CalledProcessError as e:
            stderr = ""
            if e.stderr:
                stderr = e.stderr if isinstance(e.stderr, str) else e.stderr.decode(errors="replace")
            error_msg = f"{e}\nstderr: {stderr}" if stderr else str(e)
            logger.error(f"Job {job_id} failed: {error_msg}")
            await fail_job(state, job_id, error_msg)
            clean_workspace_on_success = False
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            await fail_job(state, job_id, str(e))
            clean_workspace_on_success = False

        # Cleanup ONLY on success. Failure path leaves the workspace in place
        # so a same-container retry (legacy FastAPI path) can resume from
        # local state; the batch-job path will rebuild from object storage
        # in a fresh container anyway since the filesystem is ephemeral.
        if clean_workspace_on_success:
            try:
                shutil.rmtree(workspace, ignore_errors=True)
                logger.info(f"Job {job_id}: workspace cleaned ({workspace})")
            except Exception as e:
                logger.warning(f"Job {job_id}: workspace cleanup failed: {e}")
