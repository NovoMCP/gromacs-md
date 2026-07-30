"""Tests for trajectory analysis utility functions."""

import pytest
import tempfile
from pathlib import Path

try:
    from gromacs_md.analysis import _parse_xvg, _summarize_timeseries
except ImportError:
    pytest.skip("gromacs_md dependencies not installed", allow_module_level=True)


def test_parse_xvg_from_string():
    content = """\
# GROMACS output
@ title "RMSD"
@ xaxis "Time (ps)"
0.000 0.100
1.000 0.150
2.000 0.200
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xvg", delete=False) as f:
        f.write(content)
        f.flush()
        rows = _parse_xvg(Path(f.name))

    assert len(rows) == 3
    assert rows[0] == [0.0, 0.1]
    assert rows[2] == [2.0, 0.2]


def test_parse_xvg_missing_file():
    rows = _parse_xvg(Path("/nonexistent/file.xvg"))
    assert rows == []


def test_summarize_timeseries():
    rows = [[0, 100.0], [1, 102.0], [2, 101.0], [3, 100.5], [4, 101.5]]
    result = _summarize_timeseries(rows)

    assert result["mean"] == pytest.approx(101.0, abs=0.01)
    assert result["final"] == 101.5
    assert "std" in result
    assert "last_quarter_mean" in result
    assert isinstance(result["stable"], bool)


def test_summarize_timeseries_empty():
    assert _summarize_timeseries([]) == {}


def test_summarize_timeseries_single():
    result = _summarize_timeseries([[0, 5.0]])
    assert result["mean"] == 5.0
    assert result["final"] == 5.0
    assert result["std"] == 0.0
