"""Tests for thinning the job result's time series."""

import json

from gromacs_md.result_compact import compact_analysis, thin_series, write_full_analysis


def _series(n):
    return [[float(i), float(i) * 0.1] for i in range(n)]


def test_thin_series_keeps_ends_and_count():
    pts = _series(5001)
    thinned = thin_series(pts, 200)
    assert len(thinned) == 200
    assert thinned[0] == pts[0] and thinned[-1] == pts[-1]
    times = [p[0] for p in thinned]
    assert times == sorted(times)


def test_short_series_untouched():
    pts = _series(50)
    assert thin_series(pts, 200) is pts


def test_compact_analysis_thins_nested_series_and_records_full_count():
    analysis = {
        "equilibration": {"production_temperature": {"mean": 300.0, "trajectory": _series(5001)}},
        "rmsd": {"mean_nm": 0.14, "trajectory": _series(5001)},
        "ligand_dynamics": {"ligand_rmsd": {"mean_angstrom": 4.2, "trajectory": _series(5001)}},
        "mmgbsa": {"delta_g_kcal_mol": -46.9, "trajectory": _series(50)},
        "quality_report": {"overall": "PASS"},
    }
    out = compact_analysis(analysis, max_points=200)
    assert len(out["rmsd"]["trajectory"]) == 200
    assert out["rmsd"]["trajectory_points_full"] == 5001
    assert out["rmsd"]["mean_nm"] == 0.14
    assert len(out["equilibration"]["production_temperature"]["trajectory"]) == 200
    assert len(out["ligand_dynamics"]["ligand_rmsd"]["trajectory"]) == 200
    assert out["mmgbsa"]["trajectory"] == analysis["mmgbsa"]["trajectory"]
    assert "trajectory_points_full" not in out["mmgbsa"]
    # the input keeps its full series
    assert len(analysis["rmsd"]["trajectory"]) == 5001
    # the 1HSG-sized result drops from ~590 KB to a few tens of KB
    assert len(json.dumps(out)) < 50_000


def test_summaries_come_before_series():
    analysis = {"rmsd": {}, "equilibration": {}, "membrane": {}, "mmgbsa": {}, "quality_report": {}}
    keys = list(compact_analysis(analysis))
    assert keys[:2] == ["quality_report", "mmgbsa"]
    assert keys.index("equilibration") > keys.index("rmsd")
    assert "membrane" in keys


def test_write_full_analysis(tmp_path):
    name = write_full_analysis(tmp_path, "x", {"rmsd": {"trajectory": _series(10)}})
    assert name == "x_analysis_full.json"
    assert len(json.loads((tmp_path / name).read_text())["rmsd"]["trajectory"]) == 10
