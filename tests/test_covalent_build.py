"""Tests for the adduct residue's charges, names and junction parameters."""

import os
from pathlib import Path

import pytest

from gromacs_md import covalent_build as cb

LEAP_LOG = """\
/usr/local/bin/teLeap: Error!
Could not find bond parameter for atom types: c3 - CX
Could not find angle parameter for atom types: c3 - CX - N
Could not find angle parameter for atom types: H1 - CX - c3
 ** No torsion terms for atom types: ss-c3-CX-N
 ** No torsion terms for atom types: ss-c3-CX-N
 ** No torsion terms for atom types: h1-c3-CX-H1
"""


def test_missing_parameters_are_read_from_the_tleap_log():
    m = cb.missing_parameters(LEAP_LOG)
    assert m["bond"] == [("c3", "CX")]
    assert m["angle"] == [("c3", "CX", "N"), ("H1", "CX", "c3")]
    assert m["dihedral"] == [("ss", "c3", "CX", "N"), ("h1", "c3", "CX", "H1")]  # duplicate dropped


def test_backbone_types_map_to_gaff2():
    assert [cb._gaff2(t) for t in ("CX", "N", "H1", "C", "c3", "ss")] == ["c3", "n", "h1", "c", "c3", "ss"]
    with pytest.raises(cb.AmberBuildError):
        cb._gaff2("ZZ")


GAFF2 = Path(os.environ.get("AMBERHOME", "/opt/micromamba/envs/amber")) / "dat/leap/parm/gaff2.dat"


@pytest.mark.skipif(not GAFF2.exists(), reason="AmberTools not installed")
def test_junction_frcmod_takes_gaff2_terms(tmp_path):
    pytest.importorskip("parmed")
    path = tmp_path / "j.frcmod"
    n = cb.junction_frcmod(path, cb.missing_parameters(LEAP_LOG), [GAFF2])
    text = path.read_text()
    assert n == 5
    assert text.splitlines()[1] == "MASS"  # tleap rejects NONBON without MASS
    assert "c3-CX " in text and "ss-c3-CX-N " in text


def test_residue_charges_keep_ff14sb_backbone_and_sum_to_the_ligand_charge():
    residue = {0: "C1", 1: "N1", 2: "N", 3: "CA", 4: "C", 5: "O", 6: "CB", 7: "SG",
               8: "H", 9: "HA", 10: "HB2", 11: "HB3", 12: "H01"}
    fragment = [("x", q) for q in (0.1, -0.5, -0.4, 0.05, 0.6, -0.6, -0.1, -0.2,
                                   0.3, 0.1, 0.1, 0.1, 0.45)]
    for net in (0, 1):
        q = cb._residue_charges(fragment, residue, net)
        assert sum(q.values()) == pytest.approx(net, abs=1e-6)
        assert q[2] == cb.BACKBONE["N"][1] and q[3] == cb.BACKBONE["CA"][1]
    # the non-backbone atoms keep their BCC differences
    q = cb._residue_charges(fragment, residue, 0)
    assert q[0] - q[1] == pytest.approx(0.6, abs=1e-6)


def test_substitute_adduct_replaces_the_cysteine():
    from gromacs_md.amber_build import PreparedProtein
    pdb = (
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  N   CYS A   2       1.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      3  SG  CYS A   2       2.000   0.000   0.000  1.00  0.00           S\n"
        "ATOM      4  N   GLY A   3       3.000   0.000   0.000  1.00  0.00           N\n"
        "TER\nEND\n")
    prot = PreparedProtein(pdb, [(2, 3)], [("A", 10), ("A", 11), ("A", 12)])
    add = cb.Adduct(Path("x.lib"), [], [("N", "N", (1.0, 0, 0)), ("SG", "S", (2.0, 0, 0)),
                                        ("C9", "C", (3.0, 0, 0))], ["C9"], "C9", "bonded_form", 0)
    out = cb.substitute_adduct(prot, "A", 11, add)
    lines = [l for l in out.pdb_text.splitlines() if l.startswith("ATOM")]
    assert [l[17:20] for l in lines] == ["ALA", "CVX", "CVX", "CVX", "GLY"]
    assert out.disulfides == []  # the bonded cysteine can't be in a disulfide
    with pytest.raises(cb.CovalentLigandError, match="isn't in the cleaned protein"):
        cb.substitute_adduct(prot, "B", 11, add)
