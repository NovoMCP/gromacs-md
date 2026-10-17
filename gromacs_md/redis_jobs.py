"""Redis job tracking — orchestrator-compatible 3-key format."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Any

from .config import JOB_HEARTBEAT_SECONDS, JOB_STALE_SECONDS
from .state import AppState

if TYPE_CHECKING:
    from intake import RoutingDecision

logger = logging.getLogger(__name__)


async def update_job_status(state: AppState, job_id: str, status: str, progress: Dict[str, Any]):
    """Update job status in Redis using the orchestrator-compatible format."""
    if not state.redis:
        return
    try:
        key = f"novomcp:job:{job_id}"
        data = {
            "job_id": job_id,
            "status": status,
            "progress": json.dumps(progress),
            "last_updated": datetime.utcnow().isoformat(),
        }
        await state.redis.hset(key, mapping=data)
        await state.redis.expire(key, 86400)
        logger.info(f"Job {job_id}: {progress.get('percentage', 0)}% - {progress.get('message', '')}")
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")


async def complete_job(state: AppState, job_id: str, result: Dict[str, Any]):
    """Mark job as completed in Redis (3-key format for orchestrator compatibility)."""
    if not state.redis:
        return
    try:
        now = datetime.utcnow().isoformat()
        progress = {"percentage": 100, "message": "Simulation completed", "step": "completed"}

        unified_key = f"novomcp:job:{job_id}"
        await state.redis.hset(unified_key, mapping={
            "job_id": job_id,
            "status": "completed",
            "completed_at": now,
            "result": json.dumps(result),
            "progress": json.dumps(progress),
            "last_updated": now,
        })
        await state.redis.expire(unified_key, 604800)

        result_key = f"novomcp:job_result:{job_id}"
        await state.redis.set(result_key, json.dumps(result), ex=604800)

        cache_key = f"novomcp:cache:jobs:{job_id}"
        cache_data = {
            "job_id": job_id,
            "status": "completed",
            "completed_at": now,
            "result": result,
            "progress": progress,
            "last_updated": now,
        }
        await state.redis.set(cache_key, json.dumps(cache_data), ex=604800)

        logger.info(f"Job {job_id} completed")
    except Exception as e:
        logger.error(f"Failed to complete job {job_id}: {e}")


async def fail_job(state: AppState, job_id: str, error: str):
    """Mark job as failed in Redis."""
    if not state.redis:
        return
    try:
        await update_job_status(state, job_id, "failed", {
            "percentage": 0,
            "message": f"Simulation failed: {error}",
            "step": "failed",
            "error": error,
        })
    except Exception as e:
        logger.error(f"Failed to mark job {job_id} as failed: {e}")


async def heartbeat(state: AppState, job_id: str):
    """Record that a worker is still on this job."""
    if not state.redis:
        return
    try:
        await state.redis.hset(f"novomcp:job:{job_id}", "heartbeat_at", datetime.utcnow().isoformat())
    except Exception as e:
        logger.warning(f"Heartbeat for {job_id} failed: {e}")


async def keep_alive(state: AppState, job_id: str, interval: float = JOB_HEARTBEAT_SECONDS):
    """Heartbeat until cancelled. Run alongside a job for its whole life."""
    while True:
        await heartbeat(state, job_id)
        await asyncio.sleep(interval)


def heartbeat_age_seconds(data: Dict[str, Any]) -> float | None:
    """Seconds since the job's last heartbeat, or None for jobs without one."""
    try:
        return (datetime.utcnow() - datetime.fromisoformat(data["heartbeat_at"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


async def fail_if_orphaned(state: AppState, job_id: str, data: Dict[str, Any]) -> bool:
    """Fail an unfinished job whose worker stopped heartbeating. True if it did."""
    if data.get("status") not in ("queued", "running", "processing"):
        return False
    age = heartbeat_age_seconds(data)
    if age is None or age < JOB_STALE_SECONDS:
        return False
    await fail_job(
        state, job_id,
        f"gromacs-md stopped working on this job (no heartbeat for {int(age // 60)} min; "
        "the service was restarted or crashed). Resubmit it.",
    )
    return True


async def refuse_job(state: AppState, job_id: str, decision: RoutingDecision):
    """Mark a job as refused by the intake classifier."""
    primary_reason = decision.reasons[0] if decision.reasons else "system unsupported"
    result = {
        "status": "refused",
        "primary_reason": primary_reason,
        "reasons": decision.reasons,
        "suggested_branch": decision.suggested_branch,
        "profile": decision.profile.model_dump(),
    }
    if state.redis:
        try:
            now = datetime.utcnow().isoformat()
            progress = {
                "percentage": 100,
                "message": f"Refused: {primary_reason}",
                "step": "refused",
            }
            unified_key = f"novomcp:job:{job_id}"
            await state.redis.hset(unified_key, mapping={
                "job_id": job_id,
                "status": "refused",
                "completed_at": now,
                "result": json.dumps(result),
                "progress": json.dumps(progress),
                "last_updated": now,
            })
            await state.redis.expire(unified_key, 604800)

            result_key = f"novomcp:job_result:{job_id}"
            await state.redis.set(result_key, json.dumps(result), ex=604800)

            cache_key = f"novomcp:cache:jobs:{job_id}"
            cache_data = {
                "job_id": job_id,
                "status": "refused",
                "completed_at": now,
                "result": result,
                "progress": progress,
                "last_updated": now,
            }
            await state.redis.set(cache_key, json.dumps(cache_data), ex=604800)
        except Exception as e:
            logger.error(f"Failed to record refusal for {job_id}: {e}")
    logger.info(
        f"Job {job_id} refused: {primary_reason} "
        f"(suggested_branch={decision.suggested_branch})"
    )
