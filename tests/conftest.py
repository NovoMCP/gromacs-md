"""Shared fixtures for the gromacs-md test suite."""

import asyncio
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from gromacs_md.state import AppState


@pytest.fixture
def mock_redis():
    """Mock async Redis client."""
    r = AsyncMock()
    r.ping = AsyncMock(return_value=True)
    r.hset = AsyncMock()
    r.hgetall = AsyncMock(return_value={})
    r.set = AsyncMock()
    r.get = AsyncMock(return_value=None)
    r.expire = AsyncMock()
    r.delete = AsyncMock()
    return r


@pytest.fixture
def mock_blob():
    """Mock S3 storage client."""
    blob = MagicMock()
    container = MagicMock()
    blob.get_container_client.return_value = container
    container.upload_blob = MagicMock()
    return blob


@pytest.fixture
def mock_state(mock_redis, mock_blob):
    """AppState with mocked dependencies."""
    return AppState(
        redis=mock_redis,
        blob=mock_blob,
        gpu_sem=asyncio.Semaphore(3),
    )


@pytest.fixture
def empty_state():
    """AppState with no external services (redis=None, blob=None)."""
    return AppState(
        redis=None,
        blob=None,
        gpu_sem=asyncio.Semaphore(3),
    )


SAMPLE_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00  0.00           C
ATOM      4  O   ALA A   1       4.000   5.000   6.000  1.00  0.00           O
HETATM    5  ZN  ZN  A 100      10.000  10.000  10.000  1.00  0.00          ZN
TER
END
"""

SAMPLE_XVG = """\
# GROMACS output
@ title "RMSD"
@ xaxis "Time (ps)"
@ yaxis "RMSD (nm)"
0.000 0.100
1.000 0.150
2.000 0.200
3.000 0.180
4.000 0.190
"""
