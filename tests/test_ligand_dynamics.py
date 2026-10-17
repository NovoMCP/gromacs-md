"""Tests for the ligand pose RMSD in analyze_ligand_dynamics."""

import subprocess

from gromacs_md import analysis


def test_ligand_rmsd_fits_the_protein_on_a_centered_trajectory(tmp_path, monkeypatch):
    (tmp_path / "x_md.gro").write_text(
        "x\n    2\n    1ALA     CA    1   0.000   0.000   0.000\n"
        "    2UNL     C1    2   0.300   0.000   0.000\n   3.0 3.0 3.0\n")
    calls = []

    def fake_run(args, input=None, **kwargs):
        calls.append((args, input))
        if args[:2] == ["gmx", "rms"] and "x_ligand_rmsd.xvg" in args:
            (tmp_path / "x_ligand_rmsd.xvg").write_text("0.0 0.0\n1.0 0.25\n")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(analysis.subprocess, "run", fake_run)
    out = analysis.analyze_ligand_dynamics(tmp_path, "x")

    trjconv = next(c for c in calls if c[0][:2] == ["gmx", "trjconv"])
    rms = next(c for c in calls if c[0][:2] == ["gmx", "rms"] and "x_ligand_rmsd.xvg" in c[0])
    # center on the protein (group 1), write the whole system (0), molecules whole
    assert trjconv[1] == b"1\n0\n" and trjconv[0][trjconv[0].index("-pbc") + 1] == "mol"
    # RMSD of the ligand (2) after a least-squares fit on the protein (1)
    assert rms[1] == b"1\n2\n"
    assert rms[0][rms[0].index("-f") + 1] == "x_md_centered.xtc"
    assert out["ligand_rmsd"]["fit"] == "protein"
    assert out["ligand_rmsd"]["final_nm"] == 0.25
