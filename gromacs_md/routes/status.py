"""GET /status/{job_id}, /results/{job_id}, /jobs/{job_id}/status — job polling."""

import json
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from ..state import AppState, get_state
from ..queue import get_gpu_queue_info

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status/{job_id}")
async def get_job_status(job_id: str, state: AppState = Depends(get_state)):
    """Get simulation job status from Redis."""
    if not state.redis:
        return {"job_id": job_id, "status": "unknown", "message": "Redis not available"}

    try:
        key = f"novomcp:job:{job_id}"
        data = await state.redis.hgetall(key)

        if not data:
            return {"job_id": job_id, "status": "not_found", "progress": 0}

        progress = {}
        if "progress" in data:
            try:
                progress = json.loads(data["progress"])
            except (json.JSONDecodeError, TypeError):
                pass

        status = data.get("status", "unknown")
        response = {
            "job_id": job_id,
            "status": status,
            "progress": progress.get("percentage", 0),
            "message": progress.get("message", ""),
            "step": progress.get("step", ""),
            "last_updated": data.get("last_updated"),
        }

        # Terminal states (completed, refused) carry their full result
        # payload inline so clients that only poll /status/ still see
        # the outcome — especially important for refused jobs whose
        # SystemProfile + reasons + suggested_branch live in `result`.
        if status in ("completed", "refused") and "result" in data:
            try:
                response["result"] = json.loads(data["result"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Compute estimated remaining time for running jobs
        if status in ("queued", "running", "processing"):
            submitted_at = data.get("submitted_at")
            est_total = data.get("estimated_runtime_minutes")
            if submitted_at and est_total:
                try:
                    elapsed = (datetime.utcnow() - datetime.fromisoformat(submitted_at)).total_seconds() / 60
                    remaining = max(0, round(int(est_total) - elapsed))
                    response["estimated_remaining_minutes"] = remaining
                    response["estimated_total_minutes"] = int(est_total)
                    response["elapsed_minutes"] = round(elapsed)
                except (ValueError, TypeError):
                    pass
            # GPU queue info
            response["gpu_queue"] = get_gpu_queue_info(state)

        return response
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return {"job_id": job_id, "status": "error", "message": str(e)}


@router.get("/results/{job_id}")
async def get_job_results(job_id: str, state: AppState = Depends(get_state)):
    """Get simulation results from Redis.

    Treats both `completed` and `refused` as terminal "has result"
    states. Refused jobs carry a structured `SystemProfile` + reasons
    + suggested_branch in the same `result` field that completed jobs
    use — the classifier refusal is a successful outcome, not an error.
    """
    if not state.redis:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        # Try unified hash first
        key = f"novomcp:job:{job_id}"
        data = await state.redis.hgetall(key)

        if data and data.get("status") in ("completed", "refused"):
            terminal_status = data.get("status")
            result = {}
            if "result" in data:
                try:
                    result = json.loads(data["result"])
                except (json.JSONDecodeError, TypeError):
                    pass
            return {"job_id": job_id, "status": terminal_status, "result": result}

        # Try legacy result key (completed jobs only — refused jobs are
        # always written through the unified hash path above).
        result_key = f"novomcp:job_result:{job_id}"
        result_str = await state.redis.get(result_key)
        if result_str:
            return {"job_id": job_id, "status": "completed", "result": json.loads(result_str)}

        if data:
            status = data.get("status", "unknown")
            progress = {}
            if "progress" in data:
                try:
                    progress = json.loads(data["progress"])
                except (json.JSONDecodeError, TypeError):
                    pass
            return {"job_id": job_id, "status": status, "progress": progress}

        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Backward-compatible alias
@router.get("/jobs/{job_id}/status")
async def get_job_status_alias(job_id: str, state: AppState = Depends(get_state)):
    return await get_job_status(job_id, state)
