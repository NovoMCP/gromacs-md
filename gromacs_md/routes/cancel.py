"""POST /cancel/{job_id} — stop a queued or running job."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..auth import validate_api_key
from ..redis_jobs import update_job_status
from ..state import AppState, get_state
from ..subprocess_utils import cancel_workspace

logger = logging.getLogger(__name__)

router = APIRouter()

TERMINAL = ("completed", "failed", "refused", "cancelled")


@router.post("/cancel/{job_id}", dependencies=[Depends(validate_api_key)])
async def cancel_job(job_id: str, state: AppState = Depends(get_state)):
    """Cancel a job: a queued one never starts, a running one has its
    processes killed (mdrun, ACPYPE, gmx_MMPBSA) and its workspace removed.
    Cancelling a finished job changes nothing and returns its status."""
    if not state.redis:
        raise HTTPException(status_code=503, detail="Redis not available")
    data = await state.redis.hgetall(f"novomcp:job:{job_id}")
    if not data:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    status = data.get("status")
    if status in TERMINAL:
        return {"job_id": job_id, "status": status, "cancelled": status == "cancelled",
                "message": f"Job is already {status}; nothing to cancel."}

    await update_job_status(state, job_id, "cancelled", {
        "percentage": 0, "message": "Cancelled by user", "step": "cancelled",
    })
    workspace = state.job_workspaces.get(job_id)
    killed = cancel_workspace(workspace) if workspace else 0
    logger.info(f"Job {job_id}: cancelled ({killed} processes killed)")
    return {"job_id": job_id, "status": "cancelled", "cancelled": True, "killed_processes": killed}
