"""Tests for the GPU mdrun retries and CPU rerun."""

import os
import subprocess
import time

import pytest

from gromacs_md import subprocess_utils as su

GPU_NVT = ["gmx", "mdrun", "-v", "-deffnm", "x_nvt", "-nb", "gpu", "-update", "cpu",
           "-gpu_id", "0", "-ntmpi", "1", "-ntomp", "8"]


def test_cpu_args_move_every_task_off_the_gpu(tmp_path):
    out = su.mdrun_cpu_args(GPU_NVT + ["-pme", "gpu"], tmp_path, time.time())
    assert out[out.index("-nb") + 1] == "cpu"
    assert out[out.index("-pme") + 1] == "cpu"
    assert out[out.index("-update") + 1] == "cpu"
    assert "-gpu_id" not in out
    assert out[out.index("-ntomp") + 1] == str(su.MDRUN_CPU_NTOMP)
    assert "-cpi" not in out


def test_cpu_args_resume_from_a_checkpoint_the_failed_run_wrote(tmp_path):
    started = time.time() - 1
    (tmp_path / "x_nvt.cpt").write_text("")
    out = su.mdrun_cpu_args(GPU_NVT, tmp_path, started)
    assert out[-3:] == ["-cpi", "x_nvt.cpt", "-append"]


def test_a_resumed_run_drops_nsteps(tmp_path):
    # mdrun counts -nsteps from the checkpoint's step: kept, a stage resumed
    # at step 3200 of 30000 ran to 33200
    started = time.time() - 1
    (tmp_path / "x_md.cpt").write_text("")
    prod = ["gmx", "mdrun", "-deffnm", "x_md", "-nb", "gpu", "-nsteps", "30000"]
    out = su._resume_args(prod, tmp_path, started)
    assert "-nsteps" not in out and "30000" not in out
    assert out[-3:] == ["-cpi", "x_md.cpt", "-append"]


def test_a_fresh_rerun_keeps_nsteps(tmp_path):
    prod = ["gmx", "mdrun", "-deffnm", "x_md", "-nb", "gpu", "-nsteps", "30000"]
    out = su.mdrun_cpu_args(prod, tmp_path, time.time())
    assert out[out.index("-nsteps") + 1] == "30000"


def test_cpu_args_ignore_a_stale_checkpoint(tmp_path):
    cpt = tmp_path / "x_nvt.cpt"
    cpt.write_text("")
    os.utime(cpt, (time.time() - 3600, time.time() - 3600))
    assert "-cpi" not in su.mdrun_cpu_args(GPU_NVT, tmp_path, time.time())


def _fake_runner(fail_first_n):
    calls = []

    def fake_run(args, workspace, input_data=None):
        calls.append(list(args))
        if len(calls) <= fail_first_n:
            raise subprocess.CalledProcessError(139, args)

    return calls, fake_run


def test_a_gpu_retry_that_succeeds_stays_on_the_gpu(tmp_path, monkeypatch):
    calls, fake_run = _fake_runner(fail_first_n=1)
    monkeypatch.setattr(su, "_run", fake_run)
    monkeypatch.setattr(su, "MDRUN_GPU_RETRIES", 2)
    su.run_subprocess(GPU_NVT, tmp_path)
    assert len(calls) == 2 and all(c[c.index("-nb") + 1] == "gpu" for c in calls)
    assert su.load_mdrun_retries(tmp_path) == [
        {"stage": "x_nvt", "gpu_exit_codes": [139], "outcome": "recovered_on_gpu_retry"}]


def test_after_the_gpu_retries_it_reruns_on_cpu(tmp_path, monkeypatch):
    calls, fake_run = _fake_runner(fail_first_n=3)
    monkeypatch.setattr(su, "_run", fake_run)
    monkeypatch.setattr(su, "MDRUN_GPU_RETRIES", 2)
    su.run_subprocess(GPU_NVT, tmp_path)
    assert len(calls) == 4 and calls[-1][calls[-1].index("-nb") + 1] == "cpu"
    assert su.load_mdrun_retries(tmp_path) == [
        {"stage": "x_nvt", "gpu_exit_codes": [139, 139, 139], "outcome": "rerun_on_cpu"}]


def test_cpu_fallback_off_fails_the_stage(tmp_path, monkeypatch):
    calls, fake_run = _fake_runner(fail_first_n=99)
    monkeypatch.setattr(su, "_run", fake_run)
    monkeypatch.setattr(su, "MDRUN_GPU_RETRIES", 0)
    monkeypatch.setattr(su, "MDRUN_CPU_FALLBACK", False)
    with pytest.raises(subprocess.CalledProcessError):
        su.run_subprocess(GPU_NVT, tmp_path)
    assert len(calls) == 1
    assert su.load_mdrun_retries(tmp_path)[0]["outcome"] == "failed"


def test_successful_gpu_mdrun_records_nothing(tmp_path, monkeypatch):
    calls, fake_run = _fake_runner(fail_first_n=0)
    monkeypatch.setattr(su, "_run", fake_run)
    su.run_subprocess(GPU_NVT, tmp_path)
    assert len(calls) == 1 and su.load_mdrun_retries(tmp_path) is None


def test_non_mdrun_commands_are_not_retried(tmp_path, monkeypatch):
    def fake_run(args, workspace, input_data=None):
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(su, "_run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        su.run_subprocess(["gmx", "grompp", "-f", "x.mdp"], tmp_path)
    assert su.load_mdrun_retries(tmp_path) is None
