"""Ligand charge state at pH 7."""

import pytest
from rdkit import Chem

from gromacs_md.ligand_input import prepare_ligand_smiles
from gromacs_md.protonation import protonate_ph7


def charge(smiles):
    return Chem.GetFormalCharge(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize("name, smiles, expected", [
    ("carazolol", "CC(C)NCC(O)COc1cccc2[nH]c3ccccc3c12", +1),
    ("propranolol", "CC(C)NCC(O)COc1cccc2ccccc12", +1),
    ("ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", -1),
    ("benzamidine", "NC(=N)c1ccccc1", +1),
    ("methylguanidine", "CNC(=N)N", +1),
    ("metformin", "CN(C)C(=N)NC(=N)N", +1),
    ("losartan", "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nn[nH]n2)cc1", -1),
    ("imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", +1),
    ("glycine", "NCC(=O)O", 0),          # zwitterion
    ("lysine", "NCCCCC(N)C(=O)O", +1),
    ("diphenhydramine", "CN(C)CCOC(c1ccccc1)c1ccccc1", +1),
    ("methanesulfonic acid", "CS(=O)(=O)O", -1),
    ("phosphate monoester", "COP(=O)(O)O", -1),
    ("acyl sulfonamide", "CC(=O)NS(=O)(=O)c1ccccc1", -1),
])
def test_ionizable_groups(name, smiles, expected):
    out, notes = protonate_ph7(smiles)
    assert charge(out) == expected, (name, out)
    assert notes and f"net charge {expected:+d}" in notes[0]


@pytest.mark.parametrize("name, smiles", [
    ("benzene", "c1ccccc1"),
    ("aniline", "Nc1ccccc1"),
    ("pyridine", "c1ccncc1"),
    ("imidazole", "c1c[nH]cn1"),
    ("acetamide", "CC(N)=O"),
    ("sulfonamide", "NS(=O)(=O)c1ccccc1"),
    ("phenol", "Oc1ccccc1"),
    ("N-methylmorpholine", "CN1CCOCC1"),
    ("trifluoroethylamine", "NCC(F)(F)F"),
    ("aminoacetonitrile", "NCC#N"),
    ("glycinamide", "NCC(N)=O"),
])
def test_groups_near_or_far_below_7_stay_neutral(name, smiles):
    out, notes = protonate_ph7(smiles)
    assert out == smiles and notes == [], name


def test_n_aryl_piperazine_protonates_its_nh_only():
    out, _ = protonate_ph7("C1CN(c2ccccc2)CCN1")  # the NH is an aliphatic amine (pKa about 9)
    assert charge(out) == +1


def test_gefitinib_keeps_its_morpholine_neutral():
    out, _ = protonate_ph7("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1")
    assert charge(out) == 0


def test_a_charged_smiles_is_taken_as_intended():
    for smiles in ("C[NH+](C)C", "CC(=O)[O-]", "CC(C)[NH2+]CC(O)COc1ccccc1"):
        assert protonate_ph7(smiles) == (smiles, [])


def test_prepare_ligand_smiles_reports_the_charge_state():
    smiles, notes = prepare_ligand_smiles("CC(C)NCC(O)COc1cccc2[nH]c3ccccc3c12")
    assert charge(smiles) == +1
    assert notes == ["charged at pH 7 (aliphatic amine): net charge +1"]
