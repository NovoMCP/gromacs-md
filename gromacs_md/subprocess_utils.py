"""Subprocess wrapper for GROMACS commands."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def run_subprocess(args, workspace, input_data=None):
    """Wrapper for subprocess calls (runs in executor). Captures stderr for debugging."""
    result = subprocess.run(
        args, input=input_data, cwd=workspace,
        capture_output=True, text=(input_data is None),
    )
    if result.returncode != 0:
        stderr = result.stderr if isinstance(result.stderr, str) else result.stderr.decode(errors="replace")
        logger.error(f"Command failed: {' '.join(str(a) for a in args)}\nstderr: {stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, args,
            output=result.stdout,
            stderr=result.stderr,
        )
