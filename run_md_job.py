#!/usr/bin/env python3
"""
MD Job Executor — runs as a batch/CLI job.

This CLI runs a single MD job to completion. gromacs-md's
`run_simulation_pipeline` is async (uses asyncio.create_task inside a
FastAPI server today), so the CLI runs an asyncio event loop via
asyncio.run() rather than calling the engine synchronously. Sync Redis
writes handle progress + queue dispatch; the engine itself keeps using
aioredis through AppState.

Required env (one of two paths):
  Path A — direct invocation (manual smoke test):
    MD_JOB_ID    caller-provided job ID (e.g. gro_20260514-...)
    MD_CONFIG    JSON: {pdb_id, compound_id, ligand_smiles?, duration_ns,
                         temperature?, pressure?, pipeline?}
  Path B — queue dispatch (the orchestrator's production path):
    (no MD_JOB_ID / MD_CONFIG set; the executor BLPOPs the dispatch
     queue and reads {job_id, config} from there.)

Optional env (all writes are best-effort):
  REDIS_URL                                 Redis cache for live progress
  MD_RESULTS_BUCKET                         S3 bucket for result + checkpoint
                                            persistence (default novomcp-md-data)
  DASHBOARD_URL + DASHBOARD_ADMIN_KEY       Reserved for future use; the
                                            executor writes to Redis only,
                                            the orchestrator bridges to SQL
                                            on each poll.

Optional runtime metadata (if present):
  CONTAINER_APP_JOB_EXECUTION_NAME
  CONTAINER_APP_REVISION_NAME
  HOSTNAME

Exit codes:
  0  success (or graceful SIGTERM after engine success)
  1  pipeline execution failed
  2  bad input / config / env
"""

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
    format="[NovoMCP] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("gromacs-md-job")


# ── Env ──────────────────────────────────────────────────────────────────────

JOB_ID = os.environ.get("MD_JOB_ID", "")
CONFIG_JSON = os.environ.get("MD_CONFIG", "")
REDIS_URL = os.getenv("REDIS_URL", "")
HOSTNAME = os.getenv("HOSTNAME", "unknown")
EXECUTION_NAME = os.getenv("CONTAINER_APP_JOB_EXECUTION_NAME", "local")
REVISION_NAME = os.getenv("CONTAINER_APP_REVISION_NAME", "unknown")
CLAIM_ID = f"{HOSTNAME}:{os.getpid()}"

# Redis dispatch queue. The orchestrator LPUSHes {job_id, config} JSON
# before dispatch; the executor BLPOPs on startup.
QUEUE_KEY = "novomcp:gromacs:job_queue"
QUEUE_BLPOP_TIMEOUT_S = 300  # 5 min — covers ARM scheduling + container cold start


# ── Sync clients (no asyncio) ────────────────────────────────────────────────
# A second async client is built later inside main_async() for the engine's
# AppState. The split: sync Redis from the CLI/heartbeat
# layer, async Redis from the engine.

_redis_sync = None


def _init_redis_sync():
    global _redis_sync
    if not REDIS_URL:
        logger.warning("REDIS_URL not set — Redis writes disabled")
        return
    try:
        import redis as redis_lib

        _redis_sync = redis_lib.from_url(REDIS_URL, decode_responses=True)
        _redis_sync.ping()
        logger.info("Redis (sync) connected")
    except Exception as e:
        logger.warning(f"Redis (sync) init failed: {e}")
        _redis_sync = None


def _redis_hset(job_id: str, mapping: dict, ttl: int = 86400):
    if _redis_sync is None:
        return
    try:
        key = f"novomcp:job:{job_id}"
        _redis_sync.hset(key, mapping=mapping)
        _redis_sync.expire(key, ttl)
    except Exception as e:
        logger.debug(f"Redis hset failed: {e}")


def _redis_set(key: str, value: str, ex: int):
    if _redis_sync is None:
        return
    try:
        _redis_sync.set(key, value, ex=ex)
    except Exception as e:
        logger.debug(f"Redis set failed: {e}")


def _structured(event: str, **fields):
    """Emit structured JSON log line. Searchable by job_id/event."""
    record = {"event": event, "execution_id": EXECUTION_NAME, **fields}
    logger.info(json.dumps(record))


# ── Status writers (single source of truth for state transitions) ───────────

def _write(
    job_id: str,
    status: str,
    percentage: int,
    message: str,
    step: str,
    result: Optional[dict] = None,
    error: Optional[str] = None,
):
    now = datetime.now(timezone.utc).isoformat()
    progress = {"percentage": percentage, "message": message, "step": step}

    redis_mapping = {
        "job_id": job_id,
        "status": status,
        "progress": json.dumps(progress),
        "last_updated": now,
        # execution_id surfaces the CURRENT container's runtime execution name.
        # Retry containers (replicaRetryLimit=1 after A100 preemption) get a
        # different execution_name than the original; cancel_job needs the
        # current one to target the right ARM stop, so we re-stamp it on
        # every status write. Falls back to "local" for manual smoke runs.
        "execution_id": EXECUTION_NAME,
    }
    if status in ("completed", "failed", "cancelled"):
        redis_mapping["completed_at"] = now
    if result is not None:
        redis_mapping["result"] = json.dumps(result)
    if error is not None:
        redis_mapping["error"] = error

    ttl = 604800 if status == "completed" else 86400
    _redis_hset(job_id, redis_mapping, ttl=ttl)

    if result is not None:
        _redis_set(f"novomcp:job_result:{job_id}", json.dumps(result), ex=604800)
        _redis_set(f"novomcp:cache:jobs:{job_id}", json.dumps(result), ex=604800)


def write_provisioning(job_id: str, message: str):
    _write(job_id, "provisioning", 1, message, step="provisioning")
    _structured("status_change", job_id=job_id, status="provisioning", message=message)


def write_failed(job_id: str, error: str):
    truncated = (error or "unknown error")[:500]
    _write(job_id, "failed", 0, f"Failed: {truncated[:200]}", step="failed", error=truncated)
    _structured("status_change", job_id=job_id, status="failed", error=truncated)


def write_cancelled(job_id: str, message: str):
    _write(job_id, "cancelled", 0, message, step="cancelled", error=message)
    _structured("status_change", job_id=job_id, status="cancelled")


# ── Job dispatch (Redis queue) ──────────────────────────────────────────────

def _pop_dispatch_from_queue(timeout_s: int = QUEUE_BLPOP_TIMEOUT_S):
    """Block on the MD dispatch queue until a message arrives.

    Returns (job_id, config) on success, (None, None) on timeout or
    malformed payload.
    """
    if _redis_sync is None:
        logger.warning("Redis unavailable — cannot pop job from dispatch queue")
        return None, None
    try:
        result = _redis_sync.blpop(QUEUE_KEY, timeout=timeout_s)
    except Exception as e:
        logger.error(f"BLPOP failed on {QUEUE_KEY}: {e}")
        return None, None
    if not result:
        logger.warning(
            f"BLPOP {QUEUE_KEY} timed out after {timeout_s}s — no job to run, exiting cleanly"
        )
        return None, None

    _, raw = result
    raw_str = raw.strip() if isinstance(raw, str) else raw.decode().strip()

    try:
        dispatch = json.loads(raw_str)
        job_id = dispatch.get("job_id")
        config = dispatch.get("config")
        if job_id and config:
            logger.info(f"Picked up dispatch from queue: job_id={job_id}")
            return job_id, config
    except (json.JSONDecodeError, AttributeError):
        pass

    logger.error(f"Queue message missing job_id or config: {raw_str[:200]}")
    return None, None


# ── Config persistence for retry-after-preemption ───────────────────────────
# When the GPU replica is preempted (BackoffLimitExceeded), the retry spawns a
# fresh container. The queue message is already consumed on the first attempt.
# We persist job_id + config to a Redis key so the retry can recover without
# the queue.

_CONFIG_PREFIX = "novomcp:gromacs:job_config"
_CONFIG_TTL = 86400  # 24h


def _persist_job_config(job_id: str, config: dict):
    """Store job config in Redis for retry recovery."""
    if _redis_sync is None:
        return
    try:
        key = f"{_CONFIG_PREFIX}:{job_id}"
        _redis_sync.set(key, json.dumps({"job_id": job_id, "config": config}), ex=_CONFIG_TTL)
        _redis_sync.set(f"{_CONFIG_PREFIX}:last_dispatched", job_id, ex=_CONFIG_TTL)
        logger.info(f"Persisted job config for retry recovery: {job_id}")
    except Exception as e:
        logger.debug(f"Config persist failed: {e}")


def _recover_persisted_config():
    """Try to recover job config after GPU preemption (retry scenario).

    The queue is empty because the first attempt consumed the message.
    Check Redis for the persisted config from the first attempt.
    """
    if _redis_sync is None:
        return None, None
    try:
        last_id = _redis_sync.get(f"{_CONFIG_PREFIX}:last_dispatched")
        if not last_id:
            return None, None
        last_id = last_id if isinstance(last_id, str) else last_id.decode()

        status_key = f"novomcp:job:{last_id}"
        status_data = _redis_sync.hget(status_key, "status")
        if status_data and status_data in ("completed", "failed"):
            logger.info(f"Last dispatched job {last_id} already {status_data}, not resuming")
            return None, None

        config_key = f"{_CONFIG_PREFIX}:{last_id}"
        raw = _redis_sync.get(config_key)
        if not raw:
            return None, None
        raw_str = raw if isinstance(raw, str) else raw.decode()
        dispatch = json.loads(raw_str)
        job_id = dispatch.get("job_id")
        config = dispatch.get("config")
        if job_id and config:
            logger.info(f"RETRY RECOVERY: resuming preempted job {job_id}")
            return job_id, config
    except Exception as e:
        logger.debug(f"Config recovery failed: {e}")
    return None, None


# ── PDB fetch ───────────────────────────────────────────────────────────────

def fetch_pdb(pdb_id: str) -> str:
    """Sync PDB fetch — runs once per job before the async engine starts."""
    import urllib.request

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    with urllib.request.urlopen(url, timeout=30) as resp:
        if resp.status != 200:
            raise RuntimeError(f"PDB {pdb_id} fetch returned HTTP {resp.status}")
        return resp.read().decode()


# ── Cancellation (SIGTERM on stop or replica-timeout) ───────────────────────

_shutdown_requested = threading.Event()
_active_job_id = {"id": None}


def _handle_sigterm(signum, frame):
    logger.warning(
        "SIGTERM received — will mark cancelled on clean exit (not if engine already failed)"
    )
    _shutdown_requested.set()
    # Don't write "cancelled" here — the engine may be mid-crash and about to
    # write a more informative "failed" status. Let main() decide based on
    # whether the pipeline succeeded, failed, or was interrupted cleanly.


# ── Main (async — gromacs-md's engine is async) ─────────────────────────────

async def main_async() -> int:
    global JOB_ID  # rebind so closures see the resolved id

    job_id = JOB_ID
    config: Optional[dict] = None

    if job_id and CONFIG_JSON:
        try:
            config = json.loads(CONFIG_JSON)
        except json.JSONDecodeError as e:
            logger.error(f"MD_CONFIG is not valid JSON: {e}")
            return 2

    signal.signal(signal.SIGTERM, _handle_sigterm)
    _init_redis_sync()

    # ── Resolve job_id + config ──────────────────────────────────────────────
    if not job_id:
        job_id, config = _pop_dispatch_from_queue()
        if job_id and config:
            _persist_job_config(job_id, config)
        elif not job_id:
            # Queue empty — maybe this is a retry after preemption.
            job_id, config = _recover_persisted_config()
            if not job_id:
                return 0

    if config is None:
        _active_job_id["id"] = job_id
        JOB_ID = job_id
        write_failed(job_id, "No config available — env vars empty and queue message had no config")
        return 2

    # Validate required fields. MD has few hard requirements
    # (ligand-only jobs allowed; protein-only jobs allowed) but at minimum
    # we need either a protein source or a ligand source.
    has_protein = bool(config.get("pdb_id") or config.get("pdb_content"))
    has_ligand = bool(config.get("ligand_smiles"))
    if not (has_protein or has_ligand):
        _active_job_id["id"] = job_id
        JOB_ID = job_id
        write_failed(
            job_id,
            "MD config requires either pdb_id/pdb_content or ligand_smiles",
        )
        return 2

    _active_job_id["id"] = job_id
    JOB_ID = job_id

    compound_id = config.get("compound_id") or (
        f"{config.get('pdb_id')}-ligand"
        if config.get("pdb_id")
        else f"mol-{job_id.split('_')[-1]}"
    )
    simulation_ns = float(config.get("duration_ns") or config.get("simulation_ns") or 10.0)
    temperature = float(config.get("temperature", 300.0))
    pressure = float(config.get("pressure", 1.0))
    pdb_id = config.get("pdb_id")
    ligand_smiles = config.get("ligand_smiles")
    intent = config.get("intent")  # scientific intent for v3 quality grading
    adaptive_equilibration = bool(config.get("adaptive_equilibration", False))

    _structured(
        "job_started",
        job_id=JOB_ID,
        revision=REVISION_NAME,
        gpu_type="A100",
        compound_id=compound_id,
        pdb_id=pdb_id,
        duration_ns=simulation_ns,
        has_protein=has_protein,
        has_ligand=has_ligand,
        # Ship 2 observability: surface scientific intent + adaptive flag in
        # the structured log so we can see in production whether they made
        # it through the MCP gateway dispatch chain.
        intent=intent,
        adaptive_equilibration=adaptive_equilibration,
    )

    # ── Blob client init ─────────────────────────────────────────────────────
    blob_client = None
    try:
        from gromacs_md.s3_checkpoint import get_blob_client

        blob_client = get_blob_client()
        if blob_client:
            logger.info("Blob checkpoint storage enabled")
    except Exception as e:
        logger.warning(f"Blob client init failed (continuing without): {e}")

    # ── Check for checkpoint resume (retry after preemption) ─────────────────
    # If a manifest exists in Blob, download all checkpoint files into a fresh
    # workdir, then pass resume_workdir + completed_stages into the pipeline.
    # The pipeline's stage-gating skips completed stages and resumes production
    # mid-trajectory via `gmx mdrun -cpi md.cpt -append`.
    resume_workdir = None
    resume_completed_stages = None
    resume_manifest = None
    if blob_client:
        try:
            from gromacs_md.s3_checkpoint import download_checkpoints
            import tempfile

            SCRATCH = os.getenv("SCRATCH_DIR", "/tmp")
            resume_dir = Path(tempfile.mkdtemp(dir=SCRATCH, prefix="md_resume_"))
            manifest = download_checkpoints(blob_client, JOB_ID, resume_dir)
            if manifest:
                resume_workdir = resume_dir
                resume_completed_stages = manifest.get("completed_stages", [])
                resume_manifest = manifest
                _structured(
                    "checkpoint_resume",
                    job_id=JOB_ID,
                    current_stage=manifest.get("current_stage"),
                    completed_stages=resume_completed_stages,
                    workdir=str(resume_dir),
                )
                logger.info(
                    f"RESUMING {JOB_ID}: completed={resume_completed_stages}, "
                    f"current_stage={manifest.get('current_stage')}, workdir={resume_dir}"
                )
        except Exception as e:
            logger.info(f"No checkpoint to resume from (normal for first attempt): {e}")

    # ── Provisioning ─────────────────────────────────────────────────────────
    write_provisioning(JOB_ID, "Container started, initializing")

    # ── PDB fetch (sync, fast) ───────────────────────────────────────────────
    pdb_content = config.get("pdb_content")
    if not pdb_content and pdb_id:
        try:
            pdb_content = fetch_pdb(pdb_id)
            _structured("pdb_fetched", job_id=JOB_ID, pdb_id=pdb_id, bytes=len(pdb_content))
        except Exception as e:
            write_failed(JOB_ID, f"PDB fetch failed: {e}")
            return 1

    # ── Build AppState for the engine ────────────────────────────────────────
    # Async Redis client matches what the FastAPI server uses; the pipeline
    # writes its own progress through this client via redis_jobs.update_job_status.
    from gromacs_md.state import AppState

    state = AppState()
    state.blob = blob_client
    state.gpu_sem = asyncio.Semaphore(1)  # Per-Job semaphore is meaningless (single GPU per replica) but harmless

    if REDIS_URL:
        try:
            import redis.asyncio as aioredis

            state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            await state.redis.ping()
            logger.info("Redis (async) connected for engine")
        except Exception as e:
            logger.warning(f"Async Redis init failed (engine will write best-effort): {e}")
            state.redis = None

    # ── Checkpoint callbacks ─────────────────────────────────────────────────
    # Fired by simulation.py after each stage's _finalize_stage; uploads to
    # Blob + rewrites the manifest. Manifest state is held in this closure
    # so each subsequent stage's manifest reflects everything completed so far.
    from gromacs_md.s3_checkpoint import (
        upload_stage_checkpoint,
        upload_manifest,
        upload_production_cpt,
    )
    from gromacs_md.manifest import (
        Manifest,
        InputParams,
        ProductionProgress,
    )

    manifest_state = {
        "completed_stages": list(resume_completed_stages or []),
        "production_progress": (
            ProductionProgress(**resume_manifest.get("production_progress", {}))
            if resume_manifest else ProductionProgress()
        ),
    }
    input_params = InputParams(
        pdb_id=pdb_id or "",
        compound_id=compound_id,
        ligand_smiles=ligand_smiles,
        duration_ns=simulation_ns,
        temperature=temperature,
        force_field=config.get("force_field", "amber99sb-ildn"),
    )

    def _build_manifest(current_stage: str) -> dict:
        m = Manifest(
            job_id=JOB_ID,
            pipeline="soluble",
            current_stage=current_stage,
            completed_stages=list(manifest_state["completed_stages"]),
            production_progress=manifest_state["production_progress"],
            input_params=input_params,
        )
        return m.model_dump()

    def _checkpoint_cb(stage: str, stage_dir: Path) -> None:
        if not blob_client:
            return
        try:
            upload_stage_checkpoint(blob_client, JOB_ID, stage, stage_dir)
            if stage not in manifest_state["completed_stages"]:
                manifest_state["completed_stages"].append(stage)
            # Manifest's current_stage = the next stage that would run.
            # For the foundation PR we just record the completed stage; the
            # retry container reads completed_stages and skips them all.
            upload_manifest(blob_client, JOB_ID, _build_manifest(current_stage=stage))
            _structured(
                "stage_complete",
                job_id=JOB_ID,
                stage=stage,
                completed_stages=list(manifest_state["completed_stages"]),
            )
        except Exception as e:
            logger.warning(f"Stage checkpoint callback failed for {stage}: {e}")

    def _production_cpt_cb(cpt_path: Path) -> None:
        if not blob_client:
            return
        try:
            upload_production_cpt(blob_client, JOB_ID, cpt_path)
            # Re-upload the manifest with current_stage=production so the
            # retry path knows the cpt was uploaded mid-stage.
            upload_manifest(blob_client, JOB_ID, _build_manifest(current_stage="production"))
        except Exception as e:
            logger.debug(f"Production cpt callback failed: {e}")

    # ── Run the pipeline ─────────────────────────────────────────────────────
    # The pipeline writes its own progress to Redis via redis_jobs.update_job_status
    # at every stage boundary. Resume wiring (resume_workdir + completed_stages
    # + callbacks) is what makes the job survive GPU preemption.
    from gromacs_md.simulation import run_simulation_pipeline

    try:
        await run_simulation_pipeline(
            state,
            job_id=JOB_ID,
            compound_id=compound_id,
            pdb_content=pdb_content,
            ligand_smiles=ligand_smiles,
            simulation_ns=simulation_ns,
            temperature=temperature,
            pressure=pressure,
            pdb_id=pdb_id,
            resume_workdir=resume_workdir,
            completed_stages=resume_completed_stages,
            checkpoint_callback=_checkpoint_cb,
            production_cpt_callback=_production_cpt_cb,
            intent=intent,
            adaptive_equilibration=adaptive_equilibration,
        )
    except Exception as e:
        logger.exception(f"Pipeline execution raised: {e}")
        write_failed(JOB_ID, str(e))
        return 1
    finally:
        if state.redis is not None:
            try:
                await state.redis.aclose()
            except Exception:
                pass

    # The pipeline writes its own completed/failed status via complete_job /
    # fail_job. Read it back to determine our exit code + decide on cleanup.
    final_status = None
    if _redis_sync is not None:
        try:
            final_status = _redis_sync.hget(f"novomcp:job:{JOB_ID}", "status")
        except Exception:
            pass

    if final_status == "failed":
        return 1

    if _shutdown_requested.is_set() and final_status not in ("failed", "completed"):
        write_cancelled(JOB_ID, "SIGTERM during pipeline execution")
        return 0

    if final_status == "completed" and blob_client:
        try:
            from gromacs_md.s3_checkpoint import delete_checkpoints

            delete_checkpoints(blob_client, JOB_ID)
        except Exception as e:
            logger.warning(f"Checkpoint cleanup failed: {e}")

    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
