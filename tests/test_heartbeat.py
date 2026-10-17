"""Tests for job heartbeats, orphaned-job detection and production progress."""

import asyncio
from datetime import datetime, timedelta

import pytest

from gromacs_md import redis_jobs, simulation
from gromacs_md.routes import status as status_route

MD_LOG = """\
           Step           Time
              0        0.00000

   Energies (kJ/mol)
...
           Step           Time
        2575000     5150.00000

   Energies (kJ/mol)
"""


def _ago(seconds):
    return (datetime.utcnow() - timedelta(seconds=seconds)).isoformat()


def test_production_ns_done_reads_the_last_energy_record(tmp_path):
    log = tmp_path / "x_md.log"
    log.write_text(MD_LOG)
    assert simulation.production_ns_done(log) == pytest.approx(5.15)
    assert simulation.production_ns_done(tmp_path / "missing.log") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("data,orphaned", [
    ({"status": "processing", "heartbeat_at": _ago(600)}, True),
    ({"status": "queued", "heartbeat_at": _ago(600)}, True),
    ({"status": "processing", "heartbeat_at": _ago(30)}, False),
    ({"status": "processing"}, False),                        # older job without heartbeats
    ({"status": "completed", "heartbeat_at": _ago(600)}, False),
])
async def test_only_unfinished_jobs_with_a_stale_heartbeat_are_failed(mock_state, monkeypatch, data, orphaned):
    failed = []

    async def fake_fail(state, job_id, msg):
        failed.append(msg)

    monkeypatch.setattr(redis_jobs, "fail_job", fake_fail)
    assert await redis_jobs.fail_if_orphaned(mock_state, "j", data) is orphaned
    assert bool(failed) is orphaned
    if orphaned:
        assert "restarted or crashed" in failed[0] and "Resubmit" in failed[0]


@pytest.mark.asyncio
async def test_pipeline_heartbeats_while_running_and_stops_after(mock_state, monkeypatch):
    beats = []

    async def fake_beat(state, job_id):
        beats.append(job_id)

    async def fake_pipeline(state, job_id, *args, **kwargs):
        await asyncio.sleep(0.05)

    monkeypatch.setattr(redis_jobs, "heartbeat", fake_beat)
    monkeypatch.setattr(simulation, "_run_simulation_pipeline", fake_pipeline)
    monkeypatch.setattr(simulation, "keep_alive", lambda s, j: redis_jobs.keep_alive(s, j, interval=0.01))
    await simulation.run_simulation_pipeline(mock_state, "j1", "x", None, "CCO", 0.01, 300.0, 1.0)
    n = len(beats)
    assert n >= 2 and set(beats) == {"j1"}
    await asyncio.sleep(0.05)
    assert len(beats) == n  # stopped with the job


@pytest.mark.asyncio
async def test_status_reports_an_orphaned_job_as_failed(mock_state):
    stale = {"status": "processing", "heartbeat_at": _ago(900), "progress": '{"percentage": 67}'}
    records = [stale]

    async def hgetall(key):
        return records[-1]

    async def hset(key, mapping=None, **kw):
        records.append({**records[-1], **(mapping or {})})

    mock_state.redis.hgetall = hgetall
    mock_state.redis.hset = hset
    out = await status_route.get_job_status("j", mock_state)
    assert out["status"] == "failed"
    assert "Resubmit" in out["message"]
