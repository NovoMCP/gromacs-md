"""GPU queue management and runtime estimation."""

from .config import MAX_CONCURRENT_SIMS
from .state import AppState


def estimate_runtime_minutes(
    simulation_ns: float,
    has_protein: bool = True,
    has_ligand: bool = False,
    concurrent_jobs: int = 0,
) -> int:
    """Estimate runtime in minutes for A100 GPU."""
    min_per_ns = 3 if concurrent_jobs <= 1 else (4 if concurrent_jobs == 2 else 5)
    production_min = max(1, round(simulation_ns * min_per_ns))

    if has_protein and has_ligand:
        overhead_min = 12
    elif has_protein:
        overhead_min = 10
    else:
        overhead_min = 6

    return production_min + overhead_min


def get_gpu_queue_info(state: AppState) -> dict:
    """Get current GPU utilization info from the semaphore."""
    if not state.gpu_sem:
        return {"active_jobs": 0, "max_concurrent": MAX_CONCURRENT_SIMS, "queue_available": MAX_CONCURRENT_SIMS}
    free = state.gpu_sem._value
    active = MAX_CONCURRENT_SIMS - free
    return {
        "active_jobs": active,
        "max_concurrent": MAX_CONCURRENT_SIMS,
        "queue_available": free,
    }
