"""Subprocess wrapper for GROMACS commands."""

import json
import logging
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional, Set

from .config import MDRUN_CPU_FALLBACK, MDRUN_CPU_NTOMP, MDRUN_GPU_RETRIES, NTOMP

logger = logging.getLogger(__name__)

# Per-workspace record of mdrun stages that needed a retry; read back into the
# job result so a retried (or slower CPU) stage is visible, not silent.
RETRY_FILE = "_mdrun_retries.json"

# mdrun options that place work on the GPU. On the CPU rerun the task options
# become "cpu" and the device-selection options are dropped.
_GPU_TASK_OPTS = ("-nb", "-pme", "-pmefft", "-bonded", "-update")
_GPU_DEVICE_OPTS = ("-gpu_id", "-gputasks")

# Simulations running now, set by main at startup from the GPU semaphore.
active_simulations: Callable[[], int] = lambda: 1

# Workspaces of cancelled jobs: their processes are killed and not retried.
_cancelled: Set[str] = set()


class JobCancelled(Exception):
    """The job was cancelled while one of its commands ran."""


def cancel_workspace(workspace) -> int:
    """Mark a job's workspace cancelled and kill every process running in it.

    Matching on the working directory, plus those processes' descendants,
    covers whatever the job is running: mdrun, ACPYPE and the sqm it starts
    in a temp dir, gmx_MMPBSA. Returns the number of processes killed.
    """
    ws = os.path.realpath(str(workspace))
    _cancelled.add(ws)
    parent: dict = {}
    targets = set()
    for pid in map(int, filter(str.isdigit, os.listdir("/proc"))):
        try:
            with open(f"/proc/{pid}/stat") as f:
                parent[pid] = int(f.read().rsplit(")", 1)[1].split()[1])
            cwd = os.path.realpath(os.readlink(f"/proc/{pid}/cwd"))
        except (OSError, IndexError, ValueError):
            continue
        if pid != os.getpid() and (cwd == ws or cwd.startswith(ws + os.sep)):
            targets.add(pid)
    # and their descendants: ACPYPE runs sqm in a temp dir outside the workspace
    grew = True
    while grew:
        children = {p for p, pp in parent.items() if pp in targets and p != os.getpid()}
        grew = not children <= targets
        targets |= children
    killed = 0
    for pid in targets:
        try:
            os.kill(pid, signal.SIGKILL)
            killed += 1
        except OSError:
            pass
    return killed


def is_cancelled(workspace) -> bool:
    return os.path.realpath(str(workspace)) in _cancelled


def forget_cancelled(workspace):
    _cancelled.discard(os.path.realpath(str(workspace)))


def cpu_rerun_threads() -> int:
    """OpenMP threads for a CPU rerun: MDRUN_CPU_NTOMP, or the cores the other
    running simulations leave free (at least NTOMP)."""
    if MDRUN_CPU_NTOMP:
        return MDRUN_CPU_NTOMP
    others = max(0, active_simulations() - 1)
    return max(NTOMP, (os.cpu_count() or NTOMP) - NTOMP * others)


def _run(args, workspace, input_data=None):
    if is_cancelled(workspace):
        raise JobCancelled(str(workspace))
    result = subprocess.run(
        args, input=input_data, cwd=workspace,
        capture_output=True, text=(input_data is None),
    )
    if result.returncode != 0 and is_cancelled(workspace):
        raise JobCancelled(str(workspace))
    if result.returncode != 0:
        stderr = result.stderr if isinstance(result.stderr, str) else result.stderr.decode(errors="replace")
        logger.error(f"Command failed: {' '.join(str(a) for a in args)}\nstderr: {stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, args,
            output=result.stdout,
            stderr=result.stderr,
        )


def _opt(args, name) -> Optional[str]:
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            return str(args[i + 1])
    return None


def _is_gpu_mdrun(args) -> bool:
    return list(args[:2]) == ["gmx", "mdrun"] and _opt(args, "-nb") == "gpu"


def _resume_args(args, workspace, failed_run_started: float) -> List[str]:
    """Continue from the checkpoint a failed run wrote, if it wrote one.

    Long production stages checkpoint every 15 min; a rerun picks up there with
    -cpi -append instead of starting the stage over. A checkpoint older than the
    failed run (or an explicit -cpi already present) is left alone.
    """
    out = [str(a) for a in args]
    deffnm = _opt(out, "-deffnm")
    if deffnm and "-cpi" not in out:
        cpt = Path(workspace) / f"{deffnm}.cpt"
        if cpt.exists() and cpt.stat().st_mtime >= failed_run_started:
            out = without_nsteps(out) + ["-cpi", cpt.name, "-append"]
    return out


def without_nsteps(args) -> List[str]:
    """Drop -nsteps for a run continued from a checkpoint.

    On a -cpi restart mdrun counts -nsteps from the checkpoint's step, so a
    stage resumed at step N would run N steps past its length. The .tpr
    already holds the stage's total.
    """
    out = [str(a) for a in args]
    if "-nsteps" in out:
        i = out.index("-nsteps")
        del out[i:i + 2]
    return out


def mdrun_cpu_args(args, workspace, failed_run_started: float) -> List[str]:
    """The CPU rerun of a failed GPU mdrun: every GPU task option becomes "cpu",
    device options are dropped and the thread count becomes cpu_rerun_threads()."""
    out: List[str] = []
    i = 0
    while i < len(args):
        a = str(args[i])
        if a in _GPU_DEVICE_OPTS:
            i += 2
            continue
        if a in _GPU_TASK_OPTS and i + 1 < len(args):
            out += [a, "cpu"]
            i += 2
            continue
        if a == "-ntomp" and i + 1 < len(args):
            out += [a, str(cpu_rerun_threads())]
            i += 2
            continue
        out.append(a)
        i += 1
    return _resume_args(out, workspace, failed_run_started)


def _record_retry(workspace, args, gpu_exit_codes, outcome):
    path = Path(workspace) / RETRY_FILE
    try:
        records = json.loads(path.read_text()) if path.exists() else []
    except ValueError:
        records = []
    records.append({"stage": _opt(args, "-deffnm"), "gpu_exit_codes": gpu_exit_codes, "outcome": outcome})
    path.write_text(json.dumps(records))


def load_mdrun_retries(workspace) -> Optional[list]:
    path = Path(workspace) / RETRY_FILE
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except ValueError:
        return None


def run_subprocess(args, workspace, input_data=None):
    """Wrapper for subprocess calls (runs in executor). Captures stderr for debugging.

    A GPU `gmx mdrun` that fails is retried on the GPU (MDRUN_GPU_RETRIES), then
    rerun once on the CPU (MDRUN_CPU_FALLBACK). On GB10 the GPU nonbonded kernel
    intermittently corrupts forces on larger protein systems — LINCS warnings,
    then a segfault or CUDA error 700 — while the same .tpr runs clean on the
    CPU. The fault is intermittent, so a GPU retry usually succeeds, and a CPU
    stage is far slower (100 ps NVT of a 60k-atom system: ~10 s GPU, ~30 min CPU).
    """
    if not _is_gpu_mdrun(args):
        return _run(args, workspace, input_data)

    exit_codes: List[int] = []
    last_error: Optional[subprocess.CalledProcessError] = None
    started = time.time()
    for attempt in range(1 + max(0, MDRUN_GPU_RETRIES)):
        run_args = args if attempt == 0 else _resume_args(args, workspace, started)
        started = time.time()
        try:
            _run(run_args, workspace, input_data)
        except subprocess.CalledProcessError as e:
            last_error = e
            exit_codes.append(e.returncode)
            logger.warning(
                f"GPU mdrun failed (exit {e.returncode}), attempt {attempt + 1} of "
                f"{1 + max(0, MDRUN_GPU_RETRIES)}: {' '.join(str(a) for a in run_args)}"
            )
            continue
        if exit_codes:
            _record_retry(workspace, args, exit_codes, "recovered_on_gpu_retry")
        return

    if not MDRUN_CPU_FALLBACK:
        _record_retry(workspace, args, exit_codes, "failed")
        raise last_error
    cpu_args = mdrun_cpu_args(args, workspace, started)
    logger.warning(f"GPU mdrun failed {len(exit_codes)}x; rerunning on the CPU: {' '.join(cpu_args)}")
    _record_retry(workspace, args, exit_codes, "rerun_on_cpu")
    return _run(cpu_args, workspace, input_data)
