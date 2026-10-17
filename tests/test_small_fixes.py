"""Tests for cancellation, CPU thread sizing, the measured estimate and the
intent / adaptive_equilibration request fields."""

import os
import subprocess
import sys
import time

import pytest
from fastapi import HTTPException

from gromacs_md import queue, redis_jobs, subprocess_utils as su
from gromacs_md.models import MDSimulationRequest
from gromacs_md.routes import cancel as cancel_route
from gromacs_md.routes import simulate


# ---- CPU threads -------------------------------------------------------------

@pytest.mark.parametrize("running,expected", [(1, 19), (2, 11), (3, 8)])
def test_cpu_rerun_takes_the_cores_other_simulations_leave(monkeypatch, running, expected):
    monkeypatch.setattr(su, "MDRUN_CPU_NTOMP", None)
    monkeypatch.setattr(su.os, "cpu_count", lambda: 19)
    monkeypatch.setattr(su, "active_simulations", lambda: running)
    assert su.cpu_rerun_threads() == expected


def test_cpu_rerun_threads_can_be_pinned(monkeypatch):
    monkeypatch.setattr(su, "MDRUN_CPU_NTOMP", 4)
    assert su.cpu_rerun_threads() == 4


def test_acpype_runs_sqm_single_threaded(tmp_path, monkeypatch):
    from gromacs_md import topology
    seen = {}

    def fake_run(args, **kwargs):
        if args[0] == "acpype":
            seen["omp"] = kwargs["env"]["OMP_NUM_THREADS"]
            d = tmp_path / "x.acpype"
            d.mkdir()
            (d / "x_GMX.gro").write_text("")
            (d / "x_GMX.itp").write_text("")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(topology.subprocess, "run", fake_run)
    topology.generate_ligand_topology(tmp_path, "x", tmp_path / "l.pdb", "CCO")
    assert seen["omp"] == "1"


# ---- cancellation ------------------------------------------------------------

def test_cancel_also_kills_children_running_elsewhere(tmp_path):
    # like ACPYPE (cwd: workspace) running sqm in a temp dir outside it
    ws = tmp_path / "md_job"
    ws.mkdir()
    elsewhere = tmp_path / "acpype_tmp"
    elsewhere.mkdir()
    code = ("import subprocess, sys, time; "
            f"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], cwd={str(elsewhere)!r}); "
            "time.sleep(60)")
    parent = subprocess.Popen([sys.executable, "-c", code], cwd=ws)
    try:
        time.sleep(0.5)
        assert su.cancel_workspace(ws) == 2
        assert parent.wait(timeout=5) == -9
    finally:
        su.forget_cancelled(ws)


def test_cancel_kills_processes_running_in_the_workspace_and_blocks_retries(tmp_path):
    ws = tmp_path / "md_job"
    ws.mkdir()
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=ws)
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], cwd=tmp_path)
    try:
        time.sleep(0.2)
        assert su.cancel_workspace(ws) == 1
        assert proc.wait(timeout=5) == -9
        assert other.poll() is None  # outside the workspace: untouched
        with pytest.raises(su.JobCancelled):
            su.run_subprocess(["gmx", "mdrun", "-deffnm", "x", "-nb", "gpu"], ws)
    finally:
        other.kill()
        su.forget_cancelled(ws)
    assert not su.is_cancelled(ws)


@pytest.mark.asyncio
async def test_a_cancelled_job_keeps_its_status(mock_state):
    store = {"status": "cancelled"}

    async def hget(key, field):
        return store.get(field)

    async def hset(key, mapping=None, **kw):
        store.update(mapping or {})

    mock_state.redis.hget = hget
    mock_state.redis.hset = hset
    await redis_jobs.update_job_status(mock_state, "j", "processing", {"percentage": 50})
    await redis_jobs.fail_job(mock_state, "j", "mdrun killed")
    await redis_jobs.complete_job(mock_state, "j", {})
    assert store == {"status": "cancelled"}


@pytest.mark.asyncio
@pytest.mark.parametrize("status,cancelled", [("processing", True), ("queued", True), ("completed", False)])
async def test_cancel_route(mock_state, monkeypatch, tmp_path, status, cancelled):
    store = {"status": status}

    async def hgetall(key):
        return dict(store)

    async def hget(key, field):
        return store.get(field)

    async def hset(key, mapping=None, **kw):
        store.update(mapping or {})

    mock_state.redis.hgetall, mock_state.redis.hget, mock_state.redis.hset = hgetall, hget, hset
    killed = []
    monkeypatch.setattr(cancel_route, "cancel_workspace", lambda ws: killed.append(ws) or 2)
    mock_state.job_workspaces["j"] = tmp_path
    out = await cancel_route.cancel_job("j", mock_state)
    assert out["cancelled"] is cancelled
    assert store["status"] == ("cancelled" if cancelled else status)
    assert killed == ([tmp_path] if cancelled else [])


# ---- estimate ----------------------------------------------------------------

def test_estimate_uses_the_measured_rate():
    # 181L on an idle GB10: ~7.2 min/ns
    assert queue.estimate_runtime_minutes(10.0, True, True, measured_min_per_ns=7.2) == 72 + 12
    assert queue.estimate_runtime_minutes(10.0, True, True) == 42  # A100 default


@pytest.mark.asyncio
async def test_measured_rates_are_kept_per_system_kind(mock_state):
    store = {}

    async def set_(key, value, ex=None):
        store[key] = value

    async def get(key):
        return store.get(key)

    mock_state.redis.set, mock_state.redis.get = set_, get
    await queue.record_min_per_ns(mock_state, "protein", 200.0)
    await queue.record_min_per_ns(mock_state, "ligand", 1440.0)
    assert await queue.measured_min_per_ns(mock_state, "protein") == 7.2
    assert await queue.measured_min_per_ns(mock_state, "ligand") == 1.0


# ---- request fields ----------------------------------------------------------

@pytest.mark.asyncio
async def test_intent_and_adaptive_equilibration_reach_the_pipeline(mock_state, monkeypatch):
    seen = {}

    async def fake_fetch(state, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(simulate, "_simulate_with_fetch", fake_fetch)
    req = MDSimulationRequest(ligand_smiles="CCO", simulation_ns=0.1, intent="pose_stability",
                              adaptive_equilibration=True)
    await simulate.submit_simulation(req, mock_state)
    await __import__("asyncio").sleep(0)
    assert seen["intent"] == "pose_stability" and seen["adaptive_equilibration"] is True


@pytest.mark.asyncio
async def test_unknown_intent_is_a_400(mock_state):
    with pytest.raises(HTTPException) as e:
        await simulate.submit_simulation(MDSimulationRequest(ligand_smiles="CCO", intent="vibes"), mock_state)
    assert e.value.status_code == 400


def test_production_rate_comes_from_mdruns_performance_line(tmp_path):
    from gromacs_md import simulation
    log = tmp_path / "x_md.log"
    log.write_text(
        "               Core t (s)   Wall t (s)        (%)\n"
        "       Time:     1234.567      154.321      800.0\n"
        "                 (ns/day)    (hour/ns)\n"
        "Performance:      198.765        0.121\n"
        "Finished mdrun on rank 0\n")
    assert simulation.production_ns_per_day(log) == 198.765
    assert simulation.production_ns_per_day(tmp_path / "none.log") is None
