"""Shared application state — initialized on startup, passed explicitly."""

from dataclasses import dataclass
from typing import Any, Optional

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


def get_state(request: Request) -> AppState:
    """FastAPI dependency — extracts AppState from request."""
    return request.app.state.deps
