#!/usr/bin/env python3
"""
GROMACS-MD Consolidated Service
GPU-enabled molecular dynamics simulation service.
Replaces: gromacs-md gateway + molecular-worker + gromacs-processor

Port 8024 - Internal service
"""

import asyncio
import logging

import os

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from intake import pfam_table_size

from gromacs_md.config import (
    PORT,
    API_KEY,
    MAX_CONCURRENT_SIMS,
    REDIS_URL,
)
from gromacs_md.state import AppState
from gromacs_md.system_info import check_gpu_available, get_gromacs_version
from gromacs_md.routes import register_routes

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GROMACS-MD Service",
    description="Consolidated GPU molecular dynamics simulation service",
    version="3.0.0",
    root_path="/gromacs-md",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_routes(app)

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    state = AppState()
    state.gpu_sem = asyncio.Semaphore(MAX_CONCURRENT_SIMS)

    # Redis
    try:
        state.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        await state.redis.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis not available: {e}. Job tracking will be in-memory only.")
        state.redis = None

    # S3 object storage. Best-effort init:
    # if boto3 is missing or the bucket isn't reachable, the service keeps
    # running and results stay local only — the same behavior the old code had
    # when object storage is not configured.
    try:
        import boto3
        bucket = os.environ.get("MD_RESULTS_BUCKET", "novomcp-md-data")
        region = os.environ.get("AWS_REGION", "us-east-1")
        client = boto3.client("s3", region_name=region)
        client.head_bucket(Bucket=bucket)
        state.blob = client
        logger.info(f"Connected to S3 (bucket: {bucket})")
    except Exception as e:
        logger.warning(f"S3 not available: {e}. Results stored locally only.")
        state.blob = None

    if not API_KEY:
        logger.warning(
            "API_KEY not set — all authenticated requests will be rejected. "
            "Set the API_KEY environment variable before deploying to production."
        )

    app.state.deps = state

    logger.info(f"GROMACS-MD v3.0.0 started on port {PORT}")
    logger.info(f"Max concurrent simulations: {MAX_CONCURRENT_SIMS}")
    logger.info(f"GPU available: {check_gpu_available()}")
    logger.info(f"GROMACS: {get_gromacs_version()}")
    logger.info(f"Intake classifier loaded ({pfam_table_size()} Pfam families)")


@app.on_event("shutdown")
async def shutdown():
    state: AppState = app.state.deps
    if state.redis:
        await state.redis.aclose()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
