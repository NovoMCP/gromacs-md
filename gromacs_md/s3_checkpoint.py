"""
S3 checkpoint management for gromacs-md.

Same stage-based layout, same best-effort semantics, same public surface
(get_blob_client/upload_stage_checkpoint/upload_production_cpt/...). The
first argument is opaque to callers, so the boto3 S3 client passes
through unchanged from the existing call sites.

S3 layout (matches the Blob version):
    s3://novomcp-md-data/md-checkpoints/{job_id}/manifest.json
    s3://novomcp-md-data/md-checkpoints/{job_id}/system/{topol.top, ...}
    s3://novomcp-md-data/md-checkpoints/{job_id}/stage_em/{state.cpt, ...}
    s3://novomcp-md-data/md-checkpoints/{job_id}/stage_production/{md.cpt, md.xtc, ...}
"""

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("gromacs-md.s3-checkpoint")

BUCKET_NAME = os.getenv("MD_CHECKPOINT_BUCKET", "novomcp-md-data")
S3_PREFIX = os.getenv("MD_CHECKPOINT_PREFIX", "md-checkpoints")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def get_s3_client():
    """Initialize a boto3 S3 client. Returns None if unreachable.

    Best-effort init: if boto3 is missing or the
    bucket isn't reachable, the executor logs a warning and runs without
    checkpoints (the Job will not survive preemption, but a single attempt
    still completes).
    """
    try:
        import boto3
    except ImportError as e:
        logger.warning(f"boto3 not available: {e}")
        return None
    try:
        client = boto3.client("s3", region_name=AWS_REGION)
        client.head_bucket(Bucket=BUCKET_NAME)
        logger.info(f"S3 checkpoint storage ready (s3://{BUCKET_NAME}/{S3_PREFIX}/)")
        return client
    except Exception as e:
        logger.warning(f"S3 storage init failed (s3://{BUCKET_NAME}): {e}")
        return None


# Backwards-compat alias so existing callers (run_md_job.py, main.py, state.py)
# keep working after the import path swap.
get_blob_client = get_s3_client


def _upload_bytes(s3_client, key: str, data: bytes):
    s3_client.put_object(Bucket=BUCKET_NAME, Key=key, Body=data)


def _upload_file(s3_client, key: str, local_path: Path):
    s3_client.upload_file(str(local_path), BUCKET_NAME, key)


def upload_stage_checkpoint(s3_client, job_id: str, stage: str, stage_dir: Path) -> bool:
    """Upload all files from a completed stage directory."""
    prefix = f"{S3_PREFIX}/{job_id}/stage_{stage}"
    if not stage_dir.is_dir():
        logger.warning(f"Stage dir does not exist, nothing to upload: {stage_dir}")
        return False
    try:
        uploaded = 0
        for f in stage_dir.iterdir():
            if f.is_file():
                _upload_file(s3_client, f"{prefix}/{f.name}", f)
                uploaded += 1
        logger.info(f"Stage checkpoint uploaded: {prefix} ({uploaded} files)")
        return True
    except Exception as e:
        logger.warning(f"Stage checkpoint upload failed for {prefix}: {e}")
        return False


def upload_production_cpt(s3_client, job_id: str, cpt_path: Path) -> bool:
    """Upload md.cpt during production (called every 5 min by the loop)."""
    if not cpt_path.is_file():
        return False
    key = f"{S3_PREFIX}/{job_id}/stage_production/md.cpt"
    try:
        _upload_file(s3_client, key, cpt_path)
        return True
    except Exception as e:
        logger.warning(f"md.cpt upload failed: {e}")
        return False


def upload_system_files(s3_client, job_id: str, workdir: Path, file_list: List[str]) -> bool:
    """Upload topology/system files needed for resume (one-time)."""
    prefix = f"{S3_PREFIX}/{job_id}/system"
    try:
        uploaded = 0
        for rel_path in file_list:
            local = workdir / rel_path
            if local.exists():
                _upload_file(s3_client, f"{prefix}/{rel_path}", local)
                uploaded += 1
        logger.info(f"System files uploaded ({uploaded} files)")
        return True
    except Exception as e:
        logger.warning(f"System file upload failed: {e}")
        return False


def upload_manifest(s3_client, job_id: str, manifest: dict) -> bool:
    """Upload JSON manifest. Written AFTER stage artifacts are durable."""
    key = f"{S3_PREFIX}/{job_id}/manifest.json"
    try:
        _upload_bytes(s3_client, key, json.dumps(manifest, indent=2).encode())
        return True
    except Exception as e:
        logger.warning(f"Manifest upload failed: {e}")
        return False


def download_checkpoints(s3_client, job_id: str, target_dir: Path) -> Optional[dict]:
    """Download all checkpoint files + manifest for a stale or preempted job."""
    prefix = f"{S3_PREFIX}/{job_id}/"
    try:
        manifest_key = f"{S3_PREFIX}/{job_id}/manifest.json"
        try:
            obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=manifest_key)
            manifest = json.loads(obj["Body"].read())
        except Exception:
            logger.warning(f"No manifest found for {job_id}")
            return None

        downloaded = 0
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            for entry in page.get("Contents", []) or []:
                key = entry["Key"]
                rel_path = key[len(prefix):]
                if rel_path == "manifest.json":
                    continue
                local_path = target_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                s3_client.download_file(BUCKET_NAME, key, str(local_path))
                downloaded += 1

        logger.info(f"Downloaded {downloaded} checkpoint files for {job_id}")
        return manifest
    except Exception as e:
        logger.error(f"Checkpoint download failed for {job_id}: {e}")
        return None


def delete_checkpoints(s3_client, job_id: str) -> None:
    """Delete all checkpoint objects for a completed job."""
    prefix = f"{S3_PREFIX}/{job_id}/"
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            for entry in page.get("Contents", []) or []:
                keys.append({"Key": entry["Key"]})
        for i in range(0, len(keys), 1000):
            chunk = keys[i : i + 1000]
            s3_client.delete_objects(
                Bucket=BUCKET_NAME, Delete={"Objects": chunk, "Quiet": True}
            )
        if keys:
            logger.info(f"Cleaned {len(keys)} checkpoint objects for {job_id}")
    except Exception as e:
        logger.warning(f"Checkpoint cleanup failed for {job_id}: {e}")


def list_system_files(workdir: Path) -> List[str]:
    """Enumerate topology/system files at the workdir root. Unchanged."""
    exts = {".top", ".itp", ".gro", ".pdb", ".mdp"}
    result = []
    for f in workdir.rglob("*"):
        if not f.is_file() or f.suffix not in exts:
            continue
        rel = f.relative_to(workdir)
        if rel.parts and rel.parts[0].startswith("stage_"):
            continue
        result.append(str(rel))
    return result
