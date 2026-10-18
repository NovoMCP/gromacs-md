"""S3 upload for MD simulation results.

Same public surface (`upload_results_to_blob`), so `simulation.py` keeps its
existing call sites. `state.blob` now holds a boto3 S3 client instead of an
The object-storage (S3) client — the field name stays for backwards compatibility
across the migration.

Layout in S3 (mirrors the old Blob path):
    s3://novomcp-md-data/gromacs-md/{job_id}/result.json
    s3://novomcp-md-data/gromacs-md/{job_id}/<output_files>
    s3://novomcp-md-data/gromacs-md/{job_id}/<*.xvg>

Without S3 (a local install), the same files are copied to
LOCAL_RESULTS_DIR/{job_id}/ when that is set, since the workspace is deleted
once the job succeeds.
"""

import json
import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# The container name from the old code corresponds to the bucket
# top-level scope. On S3 we use a bucket + prefix.
MD_RESULTS_BUCKET = os.environ.get("MD_RESULTS_BUCKET", "novomcp-md-data")
MD_RESULTS_PREFIX = os.environ.get("MD_RESULTS_PREFIX", "gromacs-md")
LOCAL_RESULTS_DIR = os.environ.get("LOCAL_RESULTS_DIR", "")
# Where LOCAL_RESULTS_DIR is on the host (it's a mounted volume): the path the
# result reports, since the container's own path means nothing to the user
RESULTS_HOST_DIR = os.environ.get("RESULTS_HOST_DIR", "")


def _result_files(workspace: Path, result: Dict):
    """Output files declared in the result, then the .xvg traces, each once."""
    seen = set()
    for name in list(result.get("output_files", [])) + sorted(p.name for p in workspace.glob("*.xvg")):
        path = workspace / name
        if name not in seen and path.exists():
            seen.add(name)
            yield path


def save_results_locally(workspace: Path, job_id: str, result: Dict,
                         results_dir: str = None) -> Optional[str]:
    """Copy result.json and the output files to <results_dir>/<job_id>/."""
    results_dir = LOCAL_RESULTS_DIR if results_dir is None else results_dir
    if not results_dir:
        return None
    dest = Path(results_dir) / job_id
    try:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "result.json").write_text(json.dumps(result, indent=2))
        for path in _result_files(workspace, result):
            shutil.copy2(path, dest / path.name)
    except OSError as e:
        logger.warning(f"Could not save results to {dest}: {e}")
        return None
    logger.info(f"Results saved to {dest}")
    return str(Path(RESULTS_HOST_DIR) / job_id) if RESULTS_HOST_DIR else str(dest)


async def upload_results_to_blob(state, workspace: Path, job_id: str, result: Dict) -> Optional[str]:
    """Upload simulation results to S3 under gromacs-md/{job_id}/.

    Returns an s3:// URL pointing at the result directory, or None if the
    client is not available. Function name preserved for backwards compatibility so
    `simulation.py` keeps its existing import path.
    """
    if not state.blob:
        local = await asyncio.get_event_loop().run_in_executor(
            None, save_results_locally, workspace, job_id, result)
        if local is None:
            logger.warning("S3 storage not available and LOCAL_RESULTS_DIR unset: "
                           "output files are deleted with the workspace")
        return local

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

    # Output files declared in the result, then the .xvg traces
    for file_path in _result_files(workspace, result):
        key = f"{prefix}/{file_path.name}"
        await loop.run_in_executor(
            None,
            lambda p=str(file_path), k=key: client.upload_file(p, MD_RESULTS_BUCKET, k),
        )

    output_location = f"s3://{MD_RESULTS_BUCKET}/{prefix}/"
    logger.info(f"Results uploaded to {output_location}")
    return output_location
