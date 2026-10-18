"""Tests for long runs: measured production ETA and the MM-GBSA frame budget."""

import asyncio

import pytest

from gromacs_md import analysis, simulation, topology
from gromacs_md.routes import status as status_route


@pytest.mark.parametrize("ns,scored", [(0.05, 3), (0.5, 26), (1.0, 46), (10.0, 50), (100.0, 50)])
def test_mmgbsa_scores_at_most_50_frames(tmp_path, ns, scored):
    topology.write_md_mdp(tmp_path, "x", ns, 300.0, 1.0)
    n = analysis._md_frame_count(tmp_path, "x")
    assert n == int(ns * 500000) // 1000 + 1  # one frame per 2 ps, plus frame 0
    interval = analysis._mmgbsa_interval(n)
    assert interval >= 10  # short runs keep every 10th frame
    assert len(range(0, n, interval)) == scored


def test_mmgbsa_interval_defaults_without_an_mdp(tmp_path):
    assert analysis._mmgbsa_interval(analysis._md_frame_count(tmp_path, "x")) == 10


def test_mmgbsa_input_carries_the_interval():
    assert "interval=100," in analysis._mmpbsa_input_with_decomp(5.0, 100)
    assert "interval=10," in analysis.MMPBSA_INPUT_GB_BASE


def _log(ps):
    return f"           Step           Time\n        {int(ps * 500)}     {ps:.5f}\n"


@pytest.mark.asyncio
async def test_production_progress_measures_the_remaining_time(tmp_path, monkeypatch):
    log = tmp_path / "x_md.log"
    updates = []

    async def fake_update(state, job_id, status, progress):
        updates.append(progress)
        # the run advances 0.5 ns between readings
        done = float(progress["message"].split(": ")[1].split()[0])
        log.write_text(_log((done + 0.5) * 1000))

    clock = iter(range(0, 10_000, 60))  # one reading per simulated minute
    monkeypatch.setattr(simulation, "update_job_status", fake_update)
    log.write_text(_log(1000.0))
    task = asyncio.create_task(simulation._production_progress(
        None, "j", log, 10.0, interval=0.001, clock=lambda: next(clock)))
    while len(updates) < 3:
        await asyncio.sleep(0.005)
    task.cancel()
    assert "production_remaining_minutes" not in updates[0]  # one reading has no rate yet
    # 1.5 ns done at the second reading: 8.5 ns left at 0.5 ns/min
    assert updates[1]["production_remaining_minutes"] == 17
    assert updates[1]["ns_per_day"] == 720.0


@pytest.mark.asyncio
async def test_status_uses_the_measured_production_time(mock_state):
    from datetime import datetime, timedelta
    data = {
        "status": "processing",
        "submitted_at": (datetime.utcnow() - timedelta(minutes=60)).isoformat(),
        "estimated_runtime_minutes": "42",
        "progress": '{"percentage": 75, "message": "Production MD (10.0ns): 4.00 ns done", '
                    '"step": "production", "production_remaining_minutes": 90, "ns_per_day": 96.0}',
    }

    async def hgetall(key):
        return data

    mock_state.redis.hgetall = hgetall
    out = await status_route.get_job_status("j", mock_state)
    assert out["estimated_remaining_minutes"] == 90 + status_route.ANALYSIS_MINUTES
    assert out["estimated_total_minutes"] == 60 + 90 + status_route.ANALYSIS_MINUTES
    assert out["ns_per_day"] == 96.0


@pytest.mark.asyncio
async def test_status_times_carry_their_zone(mock_state):
    from datetime import datetime, timedelta, timezone
    submitted = datetime.utcnow() - timedelta(minutes=60)
    data = {
        "status": "processing",
        "submitted_at": submitted.isoformat(),
        "last_updated": submitted.isoformat(),
        "estimated_runtime_minutes": "100",
        "progress": '{"percentage": 50, "message": "", "step": "production"}',
    }

    async def hgetall(key):
        return data

    mock_state.redis.hgetall = hgetall
    out = await status_route.get_job_status("j", mock_state)
    assert out["last_updated"].endswith("+00:00") and out["submitted_at"].endswith("+00:00")
    finish = datetime.fromisoformat(out["estimated_completion_at"])
    assert finish.utcoffset() == timedelta(0)
    expected = datetime.now(timezone.utc) + timedelta(minutes=out["estimated_remaining_minutes"])
    assert abs((finish - expected).total_seconds()) < 5
    assert status_route._utc("not a time") == "not a time" and status_route._utc(None) is None
