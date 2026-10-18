"""Tests for runtime estimation and GPU queue info."""

import asyncio

from gromacs_md.queue import estimate_runtime_minutes, get_gpu_queue_info
from gromacs_md.state import AppState


def test_estimate_runtime_protein_ligand():
    result = estimate_runtime_minutes(10.0, has_protein=True, has_ligand=True, concurrent_jobs=0)
    # 10ns * 3min/ns + 12min overhead = 42
    assert result == 42


def test_estimate_runtime_ligand_only():
    result = estimate_runtime_minutes(5.0, has_protein=False, has_ligand=True, concurrent_jobs=0)
    # 5ns * 3min/ns + 6min overhead = 21
    assert result == 21


def test_estimate_runtime_protein_only():
    result = estimate_runtime_minutes(1.0, has_protein=True, has_ligand=False, concurrent_jobs=0)
    # 1ns * 3min/ns + 10min overhead = 13
    assert result == 13


def test_estimate_runtime_concurrent_scaling():
    solo = estimate_runtime_minutes(100.0, concurrent_jobs=0)
    busy = estimate_runtime_minutes(100.0, concurrent_jobs=3)
    assert busy > solo


def test_gpu_queue_info_no_semaphore():
    state = AppState(redis=None, blob=None, gpu_sem=None)
    info = get_gpu_queue_info(state)
    assert info["active_jobs"] == 0
    assert info["queue_available"] == 3


def test_gpu_queue_info_with_semaphore():
    sem = asyncio.Semaphore(3)
    # Simulate 1 acquired slot
    sem._value = 2
    state = AppState(redis=None, blob=None, gpu_sem=sem)
    info = get_gpu_queue_info(state)
    assert info["active_jobs"] == 1
    assert info["queue_available"] == 2


def test_measured_overhead_replaces_the_default():
    from gromacs_md.queue import estimate_runtime_minutes
    # acetate on the GB10: ~0.6 min/ns production, ~1.4 min outside it
    assert estimate_runtime_minutes(1, has_protein=False, has_ligand=True,
                                    measured_min_per_ns=0.63, measured_overhead_minutes=1.4) == 3
    assert estimate_runtime_minutes(1, has_protein=False, has_ligand=True,
                                    measured_min_per_ns=0.63) == 7


def test_overhead_kind():
    from gromacs_md.queue import overhead_kind
    assert overhead_kind(True, True) == "protein_ligand"
    assert overhead_kind(True, False) == "protein"
    assert overhead_kind(False, True) == "ligand"
