"""S3 upload for MD simulation results.

Same public surface (`upload_results_to_blob`), so `simulation.py` keeps its
existing call sites. `state.blob` now holds a boto3 S3 client instead of an
The object-storage (S3) client — the field name stays for backwards compatibility
across the migration.

Layout in S3 (mirrors the old Blob path):
    s3://novomcp-md-data/gromacs-md/{job_id}/result.json
    s3://novomcp-md-data/gromacs-md/{job_id}/<output_files>
    s3://novomcp-md-data/gromacs-md/{job_id}/<*.xvg>
"""

import json
import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# The container name from the old code corresponds to the bucket
# top-level scope. On S3 we use a bucket + prefix.
MD_RESULTS_BUCKET = os.environ.get("MD_RESULTS_BUCKET", "novomcp-md-data")
MD_RESULTS_PREFIX = os.environ.get("MD_RESULTS_PREFIX", "gromacs-md")


async def upload_results_to_blob(state, workspace: Path, job_id: str, result: Dict) -> Optional[str]:
    """Upload simulation results to S3 under gromacs-md/{job_id}/.

    Returns an s3:// URL pointing at the result directory, or None if the
    client is not available. Function name preserved for backwards compatibility so
    `simulation.py` keeps its existing import path.
    """
    if not state.blob:
        logger.warning("S3 storage not available, skipping upload")
        return None

    loop = asyncio.get_event_loop()
    client = state.blob
    prefix = f"{MD_RESULTS_PREFIX}/{job_id}"

    # result.json
    result_key = f"{prefix}/result.json"
    body = json.dumps(result, indent=2).encode()
    await loop.run_in_executor(
        None,
        lambda: client.put_object(Bucket=MD_RESULTS_BUCKET, Key=result_key, Body=body),
    )

    # Explicit output files declared by the engine
    for filename in result.get("output_files", []):
        file_path = workspace / filename
        if not file_path.exists():
            continue
        key = f"{prefix}/{filename}"
        await loop.run_in_executor(
            None,
            lambda p=str(file_path), k=key: client.upload_file(p, MD_RESULTS_BUCKET, k),
        )

    # Energy/observable .xvg traces (analysis artifacts the engine may not list)
    for xvg_file in workspace.glob("*.xvg"):
        key = f"{prefix}/{xvg_file.name}"
        await loop.run_in_executor(
            None,
            lambda p=str(xvg_file), k=key: client.upload_file(p, MD_RESULTS_BUCKET, k),
        )

    output_location = f"s3://{MD_RESULTS_BUCKET}/{prefix}/"
    logger.info(f"Results uploaded to {output_location}")
    return output_location
