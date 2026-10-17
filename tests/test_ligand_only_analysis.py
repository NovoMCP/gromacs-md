"""Ligand-only systems: structural analysis on the ligand, not on group numbers."""

import subprocess


def test_ligand_only_analysis_uses_a_ligand_group(tmp_path, monkeypatch):
    from gromacs_md import analysis
    (tmp_path / "x_md.gro").write_text(
        "x\n    3\n    1UNL     C1    1   0.000   0.000   0.000\n"
        "    2SOL     OW    2   0.500   0.000   0.000\n    3NA      NA    3   1.000   0.000   0.000\n   3.0 3.0 3.0\n")
    calls = []

    def fake_run(args, input=None, **kwargs):
        calls.append((args[:2], args, input))
        if args[:2] == ["gmx", "rms"]:
            (tmp_path / "x_rmsd.xvg").write_text("0.0 0.0\n0.01 0.05\n")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(analysis.subprocess, "run", fake_run)
    out = analysis.analyze_trajectory(tmp_path, "x", has_ligand=True, has_protein=False)
    rms = next(c for c in calls if c[0] == ["gmx", "rms"])
    assert "-n" in rms[1] and rms[2] == b"2\n2\n"  # the Ligand group of the index
    assert not any(c[0] == ["gmx", "rmsf"] for c in calls)
    assert out["rmsd"]["group"] == "ligand" and out["rmsd"]["final_nm"] == 0.05
    assert "error" not in out

