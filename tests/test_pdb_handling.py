"""Tests for PDB cleaning and processing functions."""

import pytest

try:
    from gromacs_md.pdb_handling import clean_pdb_protein_only
except ImportError:
    pytest.skip("gromacs_md dependencies not installed", allow_module_level=True)

from tests.conftest import SAMPLE_PDB


def test_clean_pdb_strips_hetatm():
    result = clean_pdb_protein_only(SAMPLE_PDB)
    assert "HETATM" not in result
    assert "ZN" not in result


def test_clean_pdb_keeps_atoms():
    result = clean_pdb_protein_only(SAMPLE_PDB)
    assert result.count("ATOM") == 4


def test_clean_pdb_keeps_ter_end():
    result = clean_pdb_protein_only(SAMPLE_PDB)
    assert "TER" in result
    assert "END" in result


def test_clean_pdb_strips_altloc():
    pdb = """\
ATOM      1  N  AALA A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  N  BALA A   1       1.100   2.100   3.100  1.00  0.00           N
END
"""
    result = clean_pdb_protein_only(pdb)
    # Should keep altloc A, drop altloc B, and normalize A to ' '
    assert result.count("ATOM") == 1
    assert result[16] == " "  # altloc column normalized


def test_clean_pdb_raises_on_no_atoms():
    pdb = """\
HETATM    1  ZN  ZN  A 100      10.000  10.000  10.000  1.00  0.00          ZN
END
"""
    with pytest.raises(ValueError, match="no standard amino acids found"):
        clean_pdb_protein_only(pdb)
