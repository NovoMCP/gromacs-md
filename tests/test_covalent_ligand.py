"""Bonded forms of the covalent validation ligands, from their CCD entries
and crystal poses (6JXT, 5P9J, 6OIM, 7SI9)."""

import json
from pathlib import Path

import pytest

Chem = pytest.importorskip("rdkit.Chem")
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

from gromacs_md import covalent_ligand as cl  # noqa: E402

FIX = Path(__file__).parent / "fixtures" / "covalent"
POSES = json.loads((FIX / "crystal_poses.json").read_text())


def _build(code):
    p = POSES[code]
    lig = {k: tuple(v) for k, v in p["ligand"].items()}
    return cl.bonded_ligand((FIX / f"{code}.cif").read_text(), lig, p["bond_atom"],
                            cl.methylthio_cap(tuple(p["sg"]), tuple(p["cb"])))


@pytest.mark.parametrize("code, rule, formula, adduct", [
    # osimertinib: the CCD has the unreacted acrylamide; S adds to the beta carbon
    ("YY3", "michael_addition", "C29H37N7O2S", "C(=O)CCSC"),
    # ibrutinib and sotorasib: the CCD has H where the S goes
    ("8E8", "bonded_form", "C26H28N6O2S", "CSCCC(=O)N"),
    ("MOV", "bonded_form", "C31H34F2N6O3S", "CSCCC(=O)N"),
    # nirmatrelvir: thioimidate, the C=N kept
    ("4WI", "bonded_form", "C24H36F3N5O4S", "CSC(=N)"),
])
def test_bonded_forms(code, rule, formula, adduct):
    mol, got_rule, names = _build(code)
    assert got_rule == rule
    assert CalcMolFormula(mol) == formula
    assert Chem.RemoveHs(mol).HasSubstructMatch(Chem.MolFromSmarts(adduct))
    # the cap sits on the crystal SG and CB, and the ligand on its crystal pose
    conf = mol.GetConformer()
    s_idx = len(names) - 2  # SG, then CB
    assert list(conf.GetAtomPosition(s_idx)) == pytest.approx(POSES[code]["sg"])
    first = names[0]
    assert list(conf.GetAtomPosition(0)) == pytest.approx(POSES[code]["ligand"][first])


def test_michael_rule_leaves_a_plain_amide_alone():
    from rdkit import Chem as C
    mol = C.RWMol(C.MolFromSmiles("CCC(=O)N"))
    assert cl.form_bond(mol, 0) == "bonded_form"
    acryl = C.RWMol(C.MolFromSmiles("C=CC(=O)N"))
    assert cl.form_bond(acryl, 0) == "michael_addition"
    assert acryl.GetBondBetweenAtoms(0, 1).GetBondType() == C.BondType.SINGLE


def test_other_warheads():
    from rdkit import Chem as C
    nitrile = C.RWMol(C.MolFromSmiles("N#CC"))
    assert cl.form_bond(nitrile, 1) == "nitrile_to_thioimidate"
    aldehyde = C.RWMol(C.MolFromSmiles("O=CC"))
    assert cl.form_bond(aldehyde, 1) == "carbonyl_to_hemithioacetal"
    chloroacetamide = C.RWMol(C.MolFromSmiles("ClCC(=O)N"))
    assert cl.form_bond(chloroacetamide, 1) == "halide_displaced"
    assert chloroacetamide.GetNumAtoms() == 4


def test_missing_crystal_atoms_are_refused():
    p = POSES["8E8"]
    lig = {k: tuple(v) for k, v in p["ligand"].items() if k != "CAD"}
    with pytest.raises(cl.CovalentLigandError, match="CAD"):
        cl.bonded_ligand((FIX / "8E8.cif").read_text(), lig, "CAA",
                         cl.methylthio_cap(tuple(p["sg"]), tuple(p["cb"])))


def test_capped_cysteine_fragment_for_charges():
    p = POSES["YY3"]
    lig = {k: tuple(v) for k, v in p["ligand"].items()}
    side = cl.capped_cysteine(*({k: tuple(v) for k, v in p[part].items()}
                                for part in ("residue", "prev", "next")))
    mol, rule, names = cl.bonded_ligand((FIX / "YY3.cif").read_text(), lig, "C9", side)
    # ligand C28H34N7O2 (bonded) + ACE C2H3O + Cys C3H4NOS + NME CH4N
    assert CalcMolFormula(mol) == "C34H45N9O4S"
    assert Chem.GetFormalCharge(mol) == 0
    assert {"N", "CA", "SG", "C_ACE", "CH3_NME"} <= set(names)


def test_capped_cysteine_needs_the_neighbours():
    p = POSES["YY3"]
    with pytest.raises(cl.CovalentLigandError, match="CA"):
        cl.capped_cysteine({k: tuple(v) for k, v in p["residue"].items()},
                           {"C": (0, 0, 0), "O": (1, 0, 0)}, {k: tuple(v) for k, v in p["next"].items()})
