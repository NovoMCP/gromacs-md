"""GET /health — service liveness and dependency checks."""

from datetime import datetime

from fastapi import APIRouter, Depends

from ..state import AppState, get_state
from ..config import MAX_CONCURRENT_SIMS, PORT
from ..system_info import check_gpu_available, get_gromacs_version
from ..queue import get_gpu_queue_info

router = APIRouter()


@router.get("/health")
async def health_check(state: AppState = Depends(get_state)):
    redis_ok = False
    if state.redis:
        try:
            await state.redis.ping()
            redis_ok = True
        except Exception:
            pass

    return {
        "status": "healthy",
        "service": "gromacs-md",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "gpu_available": check_gpu_available(),
        "gromacs_version": get_gromacs_version(),
        "redis_connected": redis_ok,
        "blob_storage_connected": state.blob is not None,
        "max_concurrent_sims": MAX_CONCURRENT_SIMS,
        "gpu_queue": get_gpu_queue_info(state),
        "port": PORT,
    }
