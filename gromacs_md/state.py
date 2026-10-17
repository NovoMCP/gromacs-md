"""Shared application state — initialized on startup, passed explicitly."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import asyncio
import redis.asyncio as aioredis
from fastapi import Request


@dataclass
class AppState:
    redis: Optional[aioredis.Redis] = None
    # boto3 S3 client.
    # Field name preserved for backwards compatibility with simulation.py /
    # run_md_job.py call sites.
    blob: Optional[Any] = None
    gpu_sem: Optional[asyncio.Semaphore] = None
    # Running jobs' workspaces, so /cancel can stop what they're running
    job_workspaces: Dict[str, Path] = field(default_factory=dict)


def get_state(request: Request) -> AppState:
    """FastAPI dependency — extracts AppState from request."""
    return request.app.state.deps
