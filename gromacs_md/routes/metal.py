"""POST /parameterize-metal — two-phase MCPB.py metal parameterization."""

import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..state import AppState, get_state
from ..auth import validate_api_key
import os
# Object-storage bucket + prefix.
_MCPB_BUCKET = os.environ.get("MD_RESULTS_BUCKET", "novomcp-md-data")
from ..pdb_handling import fetch_pdb_from_rcsb

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/parameterize-metal", dependencies=[Depends(validate_api_key)])
async def parameterize_metal(
    pdb_id: str = Form(..., description="PDB ID of the metalloprotein"),
    metal_resid: int = Form(..., description="Residue number of the metal to parameterize"),
    qm_software: str = Form("gaussian", description="QM engine: 'gaussian' or 'orca'"),
    charge: int = Form(0, description="Total charge of the QM fragment"),
    multiplicity: int = Form(1, description="Spin multiplicity of the QM fragment"),
    qm_log: Optional[UploadFile] = File(None, description="Phase 2 (legacy single-log): Gaussian .log/.fchk or ORCA .out"),
    hessian_log: Optional[UploadFile] = File(None, description="Phase 2: the small_fc (freq) log carrying the Hessian"),
    esp_log: Optional[UploadFile] = File(None, description="Phase 2: the large_mk (Pop(MK)) log carrying the ESP charges"),
    confirmation_token: Optional[str] = Form(None, description="Token from Phase 1 linking to the workspace"),
    state: AppState = Depends(get_state),
):
    """Two-phase metal parameterization via MCPB.py.

    Phase 1 (no qm_log, no confirmation_token):
        Extracts the fragment from the PDB, runs MCPB.py step 1 to
        generate the Gaussian .com input files. Returns the .com files
        + a confirmation_token linking to the workspace. The user must
        run Gaussian/ORCA on these .com files externally.

    Phase 2 (qm_log + confirmation_token):
        Takes the user's QM .log output (generated from the Phase 1
        .com files) and runs MCPB.py steps 3-4 to extract force
        constants, fit RESP charges, and produce .frcmod + .prep +
        GROMACS topology files.

    The two-phase design respects MCPB.py's fundamental constraint:
    the QM log must correspond to the exact fragment and atom ordering
    that MCPB.py step 1 produced. You cannot mix someone else's QM
    log with a different fragment extraction.
    """
    from intake.qm_ff_bridge import (
        prepare_pdb_for_mcpb,
        extract_chain_for_mcpb,
        parameterize_naa_residues,
        extract_metal_fragment,
        generate_mcpb_input,
        resolve_metal_resid,
        run_mcpb_step1,
        run_mcpb_step34,
        convert_mcpb_to_gromacs,
        validate_qm_log,
    )

    loop = asyncio.get_event_loop()
    # compound_id intentionally deferred to after metal_resid resolution
    # in the Phase 1 branch below — if resolve_metal_resid auto-corrects
    # the resid, the compound_id and workspace prefix should reflect the
    # corrected value.
    is_phase2 = (
        (qm_log is not None or (hessian_log is not None and esp_log is not None))
        and confirmation_token is not None
    )
    # Phase 2 reuses the workspace from Phase 1, which was built with the
    # already-resolved resid. So Phase 2 doesn't need to re-resolve.
    compound_id = f"{pdb_id.upper()}-metal{metal_resid}"
    # Track any resid correction so the response surfaces it for the LLM /
    # downstream tool chain to learn from.
    metal_resid_correction = None

    try:
        # ------------------------------------------------------------------
        # Phase 2: Process QM log with the workspace from Phase 1
        # ------------------------------------------------------------------
        if is_phase2:
            # Retrieve the stashed Phase 1 workspace from blob storage
            workspace_key = f"mcpb_workspace:{confirmation_token}"
            workspace_dir = None

            if state.redis:
                try:
                    cached = await state.redis.get(workspace_key)
                    if cached:
                        workspace_dir = Path(json.loads(cached).get("workspace_dir", ""))
                except Exception:
                    pass

            if not workspace_dir or not workspace_dir.exists():
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Confirmation token '{confirmation_token}' not found or expired. "
                        f"Run Phase 1 again (call without qm_log) to generate a new workspace."
                    ),
                )

            # Save the uploaded QM log(s) into the Phase 1 workspace under the
            # names MCPB.py step 3/4 expects. MCPB.py finds logs by convention —
            # <stem>_small_fc.log (Hessian) and <stem>_large_mk.log (MK ESP),
            # matching the .com files Phase 1 emitted — NOT by the upload's
            # filename. The Hessian and ESP come from two separate Gaussian runs,
            # so Phase 2 takes two logs and writes each to its expected name.
            two_log = hessian_log is not None and esp_log is not None
            if two_log:
                def _log_target(com_glob: str) -> Optional[Path]:
                    matches = list(workspace_dir.glob(com_glob))
                    return matches[0].with_suffix(".log") if matches else None

                hessian_target = _log_target("*_small_fc.com")
                esp_target = _log_target("*_large_mk.com")
                if not hessian_target or not esp_target:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "Phase 1 workspace is missing the expected small_fc/large_mk "
                            ".com files — re-run Phase 1 to regenerate the workspace."
                        ),
                    )
                hessian_target.write_bytes(await hessian_log.read())
                esp_target.write_bytes(await esp_log.read())

                # Per-slot validation pins a wrong/swapped log up front.
                v_hess = await loop.run_in_executor(
                    None, validate_qm_log, hessian_target, qm_software, "hessian"
                )
                v_esp = await loop.run_in_executor(
                    None, validate_qm_log, esp_target, qm_software, "esp"
                )
                validation = {
                    "valid": bool(v_hess.get("valid") and v_esp.get("valid")),
                    "hessian_log": v_hess,
                    "esp_log": v_esp,
                }
                if not validation["valid"]:
                    bad = []
                    if not v_hess.get("valid"):
                        bad.append("hessian_file_id → " + "; ".join(
                            v_hess.get("warnings") or ["invalid Hessian log"]))
                    if not v_esp.get("valid"):
                        bad.append("esp_file_id → " + "; ".join(
                            v_esp.get("warnings") or ["invalid ESP log"]))
                    return {
                        "status": "invalid_qm_log",
                        "compound_id": compound_id,
                        "validation": validation,
                        "message": "QM log pair validation failed. " + " | ".join(bad),
                    }
            else:
                # Legacy single combined log (kept for back-compat / .fchk).
                qm_content = await qm_log.read()
                qm_log_path = workspace_dir / qm_log.filename
                qm_log_path.write_bytes(qm_content)
                validation = await loop.run_in_executor(
                    None, validate_qm_log, qm_log_path, qm_software
                )
                if not validation.get("valid"):
                    return {
                        "status": "invalid_qm_log",
                        "compound_id": compound_id,
                        "validation": validation,
                        "message": (
                            f"QM log validation failed. "
                            f"Sections found: {validation.get('sections_found', [])}. "
                            f"Warnings: {validation.get('warnings', [])}"
                        ),
                    }

            # Find the MCPB.py input file from Phase 1
            mcpb_inputs = list(workspace_dir.glob("*.in"))
            if not mcpb_inputs:
                raise HTTPException(status_code=500, detail="Phase 1 workspace missing .in file")
            mcpb_input = mcpb_inputs[0]

            # Run MCPB.py steps 3-4 (parameter extraction)
            try:
                outputs = await loop.run_in_executor(
                    None, run_mcpb_step34, mcpb_input, workspace_dir,
                )
            except subprocess.CalledProcessError as e:
                stderr = e.stderr if isinstance(e.stderr, str) else str(e.stderr)
                return {
                    "status": "mcpb_failed",
                    "compound_id": compound_id,
                    "error": f"MCPB.py steps 3-4 failed: {stderr[:1000]}",
                    "qm_validation": validation,
                }

            # Convert to GROMACS format if AMBER output exists
            gromacs_files = {}
            if "prmtop" in outputs and "inpcrd" in outputs:
                try:
                    gro_file, top_file = await loop.run_in_executor(
                        None, convert_mcpb_to_gromacs,
                        outputs["prmtop"], outputs["inpcrd"],
                        workspace_dir, compound_id,
                    )
                    gromacs_files = {
                        "gro": gro_file.read_text()[:10000],
                        "top_preview": top_file.read_text()[:10000],
                    }
                    # Add the GROMACS deliverables to outputs so they're zipped
                    # into blob_url and registerable as child files downstream
                    # (alongside .frcmod/.prep). Without this they'd exist only
                    # as the 10 KB previews above.
                    outputs["gro"] = gro_file
                    outputs["top"] = top_file
                except Exception as e:
                    logger.warning(f"GROMACS conversion failed: {e}")
                    gromacs_files = {"conversion_error": str(e)}

            # Read output file contents
            result_files = {}
            for key, path in outputs.items():
                if path.exists() and path.stat().st_size < 500_000:
                    result_files[key] = path.read_text(errors="replace")[:50000]

            # Upload to blob storage
            blob_url = None
            if state.blob:
                try:
                    import zipfile
                    zip_path = workspace_dir / f"{compound_id}_mcpb_output.zip"
                    with zipfile.ZipFile(zip_path, "w") as zf:
                        for key, path in outputs.items():
                            if path.exists():
                                zf.write(path, path.name)
                    s3_key = f"mcpb/{compound_id}/{zip_path.name}"
                    # state.blob is a boto3 S3 client (post-AWS port).
                    state.blob.upload_file(str(zip_path), _MCPB_BUCKET, s3_key)
                    blob_url = f"s3://{_MCPB_BUCKET}/{s3_key}"
                except Exception as e:
                    logger.warning(f"S3 upload failed for MCPB output: {e}")

            # Clean up the Redis workspace reference
            if state.redis:
                try:
                    await state.redis.delete(workspace_key)
                except Exception:
                    pass

            return {
                "status": "success",
                "phase": 2,
                "compound_id": compound_id,
                "pdb_id": pdb_id.upper(),
                "metal_resid": metal_resid,
                "qm_software": qm_software,
                "qm_validation": validation,
                "outputs": list(outputs.keys()),
                "files": result_files,
                "gromacs": gromacs_files,
                "blob_url": blob_url,
                "message": (
                    f"Phase 2 complete: metal site parameterized via MCPB.py. "
                    f"Output files: {', '.join(outputs.keys())}. "
                    f"{'GROMACS .gro/.top also generated.' if gromacs_files and 'conversion_error' not in gromacs_files else ''}"
                ),
            }

        # ------------------------------------------------------------------
        # Phase 1: Extract fragment + generate QM input files
        # ------------------------------------------------------------------
        # Use a persistent workspace (not TemporaryDirectory) so Phase 2
        # can access the same files. Cleaned up after Phase 2 or by TTL.
        workspace = Path(tempfile.mkdtemp(prefix=f"mcpb_{compound_id}_"))

        # Fetch PDB and prepare for MCPB.py
        pdb_content = await fetch_pdb_from_rcsb(pdb_id)
        raw_pdb_path = workspace / f"{pdb_id.upper()}_raw.pdb"
        raw_pdb_path.write_text(pdb_content)

        # IMPORTANT: chain extraction runs FIRST on the raw PDB, BEFORE
        # pdb4amber. The raw PDB has intact HETATM metal lines that
        # _is_metal_line() can find reliably. pdb4amber's --dry flag
        # strips metal ions as "solvent", making them invisible to
        # downstream metal detection. By extracting the chain first,
        # we guarantee the metal is present when pdb4amber runs.
        pdb_path = await loop.run_in_executor(
            None, extract_chain_for_mcpb, raw_pdb_path, metal_resid, workspace,
        )

        # pdb4amber cleans alternate conformations, non-standard names,
        # chain breaks — every MCPB.py tutorial starts with this step.
        # Runs on the single-chain PDB so it only processes one chain.
        # Metal lines are preserved/re-injected by prepare_pdb_for_mcpb
        # if --dry removes them.
        pdb_path = await loop.run_in_executor(
            None, prepare_pdb_for_mcpb, pdb_path, workspace,
        )

        # Re-read the cleaned/extracted content for metal detection
        pdb_content = pdb_path.read_text()

        # ── Post-pdb4amber metal_resid resolution ────────────────────────
        # audit_system reports the PDB-author resid (raw PDB numbering);
        # this service runs pdb4amber which renumbers residues. For 1OKL:
        # raw PDB has ZN@262 (PDB-author); post-pdb4amber it's ZN@257.
        # The user-supplied resid was correct against the raw PDB but
        # invalid downstream. Resolve against the post-pdb4amber content
        # where every subsequent step needs the corrected id. Pass the raw
        # PDB too so multi-metal proteins (e.g. 1OKL has HG + ZN) can
        # disambiguate by element. Item 2 of QM-FF-BRIDGE-COMPLETION.md.
        raw_pdb_text = raw_pdb_path.read_text(errors="replace")
        resolved_resid, correction_now = resolve_metal_resid(
            pdb_content, metal_resid, raw_pdb_content=raw_pdb_text,
        )
        if correction_now is not None:
            logger.info(
                "Auto-resolved metal_resid %d → %d post-pdb4amber for %s",
                metal_resid, resolved_resid, pdb_id,
            )
            metal_resid = resolved_resid
            metal_resid_correction = correction_now
            # Note: compound_id and workspace path keep their original
            # (pre-correction) values — they're cosmetic for log + S3
            # key purposes and renaming mid-flow would invalidate the
            # already-written raw_pdb_path / pdb_path references.

        # Detect metal element from PDB — match by RESIDUE NAME against
        # known metals, not record type. pdb4amber sometimes reclassifies
        # HETATM -> ATOM for ions, and standard amino acid ATOM records
        # can share the same resid (e.g., ALA B 129 coexists with ZN B 129).
        from intake.qm_ff_bridge import _is_metal_line
        metal_element = "ZN"  # fallback
        for line in pdb_content.splitlines():
            if not _is_metal_line(line):
                continue
            try:
                resid_str = line[22:26].strip()
                if resid_str == str(metal_resid):
                    metal_element = (line[76:78].strip() or line[12:16].strip()[:2]).upper()
                    break
            except (IndexError, ValueError):
                continue

        # Extract fragment
        fragment_pdb = await loop.run_in_executor(
            None, extract_metal_fragment,
            pdb_path, metal_resid, workspace,
        )

        # Detect and parameterize non-standard residues (NAA) near the metal.
        # MCPB.py requires mol2 + frcmod for any non-amino-acid ligand in the
        # coordination sphere (same step as the MCPB.py tutorial's antechamber
        # + parmchk2 for bound ligands like MNS in 1OKL).
        naa_mol2, naa_frcmod = await loop.run_in_executor(
            None, parameterize_naa_residues,
            pdb_path, metal_resid, metal_element, workspace,
        )
        if naa_mol2:
            logger.info(
                f"Job {compound_id}: parameterized {len(naa_mol2)} NAA residue(s) "
                f"near metal: {[p.stem for p in naa_mol2]}"
            )

        # Generate MCPB.py input (now with NAA mol2/frcmod if present)
        mcpb_input = await loop.run_in_executor(
            None, generate_mcpb_input,
            pdb_path, metal_resid, metal_element,
            qm_software, charge, multiplicity,
            workspace, compound_id,
            None,  # ligating_residues (optional)
            naa_mol2, naa_frcmod,
        )

        # Run MCPB.py step 1 (fragment extraction + .com generation)
        try:
            step1_outputs = await loop.run_in_executor(
                None, run_mcpb_step1, mcpb_input, workspace,
            )
        except subprocess.CalledProcessError as e:
            stderr = e.stderr if isinstance(e.stderr, str) else str(e.stderr)
            # Diagnostic: dump the MCPB.py input file and the metal atom
            # line from the cleaned PDB so we can debug VDW/element issues
            # without another deploy cycle.
            diag = {}
            try:
                mcpb_in_content = mcpb_input.read_text()
                diag["mcpb_input"] = mcpb_in_content
            except Exception:
                pass
            try:
                for line in pdb_content.splitlines():
                    if line.startswith(("ATOM", "HETATM")):
                        resid_str = line[22:26].strip()
                        if resid_str == str(metal_resid):
                            diag["metal_pdb_line"] = line
                            diag["metal_pdb_line_len"] = len(line)
                            diag["cols_12_16_atomname"] = line[12:16]
                            diag["cols_76_78_element"] = line[76:78] if len(line) > 77 else "(short)"
                            break
            except Exception:
                pass
            try:
                mol2_path = workspace / f"{metal_element.upper()}.mol2"
                if mol2_path.exists():
                    diag["ion_mol2"] = mol2_path.read_text()
            except Exception:
                pass
            # Clean up workspace on failure
            shutil.rmtree(workspace, ignore_errors=True)
            return {
                "status": "mcpb_step1_failed",
                "compound_id": compound_id,
                "error": f"MCPB.py step 1 failed: {stderr[:2000]}",
                "diagnostic": diag,
            }

        # Generate confirmation token and stash workspace path in Redis
        token = f"mcpb_{uuid.uuid4().hex[:16]}"
        if state.redis:
            try:
                await state.redis.set(
                    f"mcpb_workspace:{token}",
                    json.dumps({"workspace_dir": str(workspace), "compound_id": compound_id}),
                    ex=86400,  # 24h TTL — user has one day to run QM and come back
                )
            except Exception as e:
                logger.warning(f"Failed to stash workspace in Redis: {e}")

        # Read .com file contents for the response
        com_files = {}
        for key, path in step1_outputs.items():
            if path.exists() and path.suffix in (".com", ".inp", ".pdb"):
                com_files[key] = path.read_text(errors="replace")[:100000]

        # Upload .com files to blob for download
        blob_url = None
        if state.blob:
            try:
                import zipfile
                zip_path = workspace / f"{compound_id}_qm_inputs.zip"
                with zipfile.ZipFile(zip_path, "w") as zf:
                    for key, path in step1_outputs.items():
                        if path.exists():
                            zf.write(path, path.name)
                s3_key = f"mcpb/{compound_id}/{zip_path.name}"
                state.blob.upload_file(str(zip_path), _MCPB_BUCKET, s3_key)
                blob_url = f"s3://{_MCPB_BUCKET}/{s3_key}"
            except Exception as e:
                logger.warning(f"S3 upload failed for QM inputs: {e}")

        return {
            "status": "phase1_complete",
            "phase": 1,
            "compound_id": compound_id,
            "pdb_id": pdb_id.upper(),
            "metal_resid": metal_resid,
            "metal_resid_correction": metal_resid_correction,
            "metal_element": metal_element,
            "qm_software": qm_software,
            "confirmation_token": token,
            "token_expires_in_seconds": 86400,
            "qm_input_files": list(step1_outputs.keys()),
            "files": com_files,
            "blob_url": blob_url,
            "message": (
                f"Phase 1 complete: MCPB.py extracted the coordination fragment for "
                f"{metal_element}@{metal_resid} and generated Gaussian input files. "
                f"Run Gaussian on the .com files (especially the _small_fc.com and "
                f"_large_mk.com), then call again with the .log output and "
                f"confirmation_token='{token}' to produce force field parameters. "
                f"Token expires in 24 hours."
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Metal parameterization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Parameterization error: {e}")
