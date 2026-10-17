"""GPU queue management and runtime estimation."""

from typing import Optional

from .config import MAX_CONCURRENT_SIMS, MD_MIN_PER_NS
from .state import AppState

# Last production rate measured on this host (minutes per simulated ns), kept
# apart for protein systems and ligand-only ones (a few thousand atoms, many
# times faster)
MEASURED_RATE_KEY = "novomcp:gromacs:min_per_ns:{kind}"


def system_kind(has_protein: bool) -> str:
    return "protein" if has_protein else "ligand"


def estimate_runtime_minutes(
    simulation_ns: float,
    has_protein: bool = True,
    has_ligand: bool = False,
    concurrent_jobs: int = 0,
    measured_min_per_ns: Optional[float] = None,
) -> int:
    """Estimated runtime in minutes.

    Production uses the last rate measured on this host when there is one.
    Otherwise MD_MIN_PER_NS (an A100 by default), slower with more jobs on
    the GPU.
    """
    if measured_min_per_ns:
        min_per_ns = measured_min_per_ns
    else:
        min_per_ns = MD_MIN_PER_NS * (1 if concurrent_jobs <= 1 else (4 / 3 if concurrent_jobs == 2 else 5 / 3))
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


async def measured_min_per_ns(state: AppState, kind: str) -> Optional[float]:
    """The last production rate measured on this host for this kind of system."""
    if not state.redis:
        return None
    try:
        v = await state.redis.get(MEASURED_RATE_KEY.format(kind=kind))
        return float(v) if v else None
    except (ValueError, TypeError, Exception):
        return None


async def record_min_per_ns(state: AppState, kind: str, ns_per_day: float):
    """Store a measured production rate for later estimates (kept 30 days)."""
    if not state or not state.redis or ns_per_day <= 0:
        return
    try:
        await state.redis.set(MEASURED_RATE_KEY.format(kind=kind), f"{1440 / ns_per_day:.3f}", ex=30 * 86400)
    except Exception:
        pass
