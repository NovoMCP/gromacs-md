"""Tests for Redis job tracking functions."""

import json
import pytest
from unittest.mock import AsyncMock

try:
    from gromacs_md.redis_jobs import update_job_status, complete_job, fail_job
    from gromacs_md.state import AppState
except ImportError:
    pytest.skip("gromacs_md dependencies not installed", allow_module_level=True)


@pytest.mark.asyncio
async def test_update_job_status_writes_to_redis(mock_state):
    await update_job_status(mock_state, "job-1", "processing", {
        "percentage": 50, "message": "Running", "step": "production",
    })

    mock_state.redis.hset.assert_called_once()
    call_args = mock_state.redis.hset.call_args
    assert call_args[1]["mapping"]["job_id"] == "job-1"
    assert call_args[1]["mapping"]["status"] == "processing"


@pytest.mark.asyncio
async def test_update_job_status_noop_without_redis(empty_state):
    # Should not raise
    await update_job_status(empty_state, "job-1", "processing", {"percentage": 50})


@pytest.mark.asyncio
async def test_complete_job_writes_three_keys(mock_state):
    await complete_job(mock_state, "job-1", {"result": "data"})

    # Should write to unified hash + 2 legacy keys
    assert mock_state.redis.hset.call_count == 1
    assert mock_state.redis.set.call_count == 2


@pytest.mark.asyncio
async def test_fail_job_calls_update(mock_state):
    await fail_job(mock_state, "job-1", "simulation crashed")

    # fail_job delegates to update_job_status
    mock_state.redis.hset.assert_called_once()
    call_args = mock_state.redis.hset.call_args
    progress = json.loads(call_args[1]["mapping"]["progress"])
    assert "failed" in progress["step"]
