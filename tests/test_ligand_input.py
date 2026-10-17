"""Tests for checking a ligand SMILES before parameterization."""

import pytest
from fastapi import HTTPException

from gromacs_md import simulation, topology
from gromacs_md.ligand_input import LigandInputError, prepare_ligand_smiles
from gromacs_md.models import MDSimulationRequest
from gromacs_md.routes import simulate

IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CC[NH+](C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"


def test_plain_ligand_passes_unchanged():
    smiles, notes = prepare_ligand_smiles("c1ccccc1")
    assert smiles == "c1ccccc1" and notes == []


@pytest.mark.parametrize("salt,kept,charge", [
    (IMATINIB + ".CS(=O)(=O)[O-]", IMATINIB, 1),  # imatinib mesylate
    ("CC(=O)[O-].[Na+]", "CC(=O)[O-]", -1),       # sodium acetate
    ("NC(=[NH2+])c1ccccc1.[Cl-].O", "NC(=[NH2+])c1ccccc1", 1),  # salt + water
])
def test_salt_keeps_its_largest_component(salt, kept, charge):
    smiles, notes = prepare_ligand_smiles(salt)
    assert smiles == prepare_ligand_smiles(kept)[0]
    assert topology.ligand_net_charge(smiles) == charge  # not the salt's net 0
    assert len(notes) == 1 and notes[0].startswith("kept the largest component; dropped")


@pytest.mark.parametrize("smiles,element", [
    ("CC(C)C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)c1cnccn1)B(O)O", "B"),  # bortezomib
    ("[NH3][Pt]([NH3])(Cl)Cl", "Pt"),                                # cisplatin
    ("C[Si](C)(C)O", "Si"),
])
def test_elements_without_gaff2_types_are_rejected(smiles, element):
    with pytest.raises(LigandInputError, match=rf"contains {element}\b"):
        prepare_ligand_smiles(smiles)


def test_invalid_and_ambiguous_smiles_are_rejected():
    with pytest.raises(LigandInputError, match="not a valid SMILES"):
        prepare_ligand_smiles("C1CC(")
    with pytest.raises(LigandInputError, match="several molecules of the same size"):
        prepare_ligand_smiles("CCO.OCC")


@pytest.mark.asyncio
async def test_submission_with_an_unsupported_ligand_is_a_400(mock_state):
    req = MDSimulationRequest(pdb_id="1ABC", ligand_smiles="[NH3][Pt]([NH3])(Cl)Cl")
    with pytest.raises(HTTPException) as e:
        await simulate.submit_simulation(req, mock_state)
    assert e.value.status_code == 400 and "contains Pt" in e.value.detail


@pytest.mark.asyncio
async def test_pipeline_fails_the_job_with_the_reason(mock_state, monkeypatch):
    failed = {}

    async def fake_fail(state, job_id, msg):
        failed[job_id] = msg

    monkeypatch.setattr(simulation, "fail_job", fake_fail)
    await simulation.run_simulation_pipeline(mock_state, "j1", "x", None, "OB(O)c1ccccc1", 0.01, 300.0, 1.0)
    assert failed["j1"].startswith("Ligand can't be parameterized: the ligand contains B")


def test_acpype_errors_keep_only_its_messages():
    out = """==> Executing Antechamber...
ERROR: ++++++++++start_quote+++++++++++++++++
ERROR: /usr/local/bin/antechamber: line 7: /usr/local/bin/../amber.sh: No such file or directory
Cannot properly run "sqm -O -i sqm.in -o sqm.out".
ERROR: ++++++++++end_quote+++++++++++++++++++
ERROR: Antechamber failed
ERROR: Parmchk failed
ERROR: Antechamber failed
ERROR: ACPYPE FAILED: [Errno 2] No such file or directory: 'l_AC.prmtop'
Traceback (most recent call last):
"""
    assert topology.acpype_errors(out) == (
        "Antechamber failed; Parmchk failed; ACPYPE FAILED: [Errno 2] No such file or directory: 'l_AC.prmtop'")
