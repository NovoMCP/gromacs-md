"""Tests for the complex topology and the ligand's ACPYPE call."""

import subprocess

import pytest

from gromacs_md import topology

LIGAND_ITP = """\
[ moleculetype ]
;name            nrexcl
 lig              3

[ atoms ]
;   nr  type  resi  res  atom  cgnr     charge      mass
     1   ca     1   UNL    C1    1     0.000000     12.01000
"""


def _two_chain_complex(tmp_path, trailing_newline=True):
    (tmp_path / "x_protein.gro").write_text(
        "p\n    2\n    1ALA     CA    1   0.000   0.000   0.000\n"
        "    2ALA     CA    2   0.500   0.000   0.000\n   1.0 1.0 1.0\n")
    (tmp_path / "x_ligand.gro").write_text("l\n    1\n    1UNL     C1    1   0.100   0.100   0.100\n   1.0 1.0 1.0\n")
    top = ('#include "amber99sb-ildn.ff/forcefield.itp"\n'
           '#include "x_protein_Protein_chain_A.itp"\n#include "x_protein_Protein_chain_B.itp"\n'
           "[ system ]\nx\n\n[ molecules ]\n; Compound        #mols\nProtein_chain_A     1\nProtein_chain_B     1")
    (tmp_path / "x_protein.top").write_text(top + ("\n" if trailing_newline else ""))
    (tmp_path / "x_ligand.itp").write_text(LIGAND_ITP)
    _, complex_top = topology.combine_protein_ligand(
        tmp_path, "x", tmp_path / "x_protein.gro", tmp_path / "x_ligand.gro",
        tmp_path / "x_protein.top", tmp_path / "x_ligand.itp")
    text = complex_top.read_text()
    mols = [l.split()[0] for l in text.split("[ molecules ]")[1].splitlines()
            if l.strip() and not l.startswith(";")]
    return text, mols


@pytest.mark.parametrize("trailing_newline", [True, False])
def test_ligand_follows_every_protein_chain(tmp_path, trailing_newline):
    # complex_gro holds chain A, chain B, then the ligand: [ molecules ] must match
    _, mols = _two_chain_complex(tmp_path, trailing_newline)
    assert mols == ["Protein_chain_A", "Protein_chain_B", "lig"]


def test_ligand_itp_is_included_before_system(tmp_path):
    text, _ = _two_chain_complex(tmp_path)
    assert text.index('#include "x_ligand.itp"') < text.index("[ system ]")


@pytest.mark.parametrize("smiles,charge", [
    ("NC(=[NH2+])c1ccccc1", 1),
    ("[O-]C(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@H]12", -1),
    ("CCO", 0),
    ("[NH3+]CC([O-])=O", 0),
])
def test_ligand_net_charge(smiles, charge):
    assert topology.ligand_net_charge(smiles) == charge


def test_acpype_is_given_the_smiles_net_charge(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[0] == "acpype":
            d = tmp_path / "x.acpype"
            d.mkdir()
            (d / "x_GMX.gro").write_text("")
            (d / "x_GMX.itp").write_text("")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(topology.subprocess, "run", fake_run)
    topology.generate_ligand_topology(tmp_path, "x", tmp_path / "lig.pdb", "NC(=[NH2+])c1ccccc1")
    acpype = next(c for c in calls if c[0] == "acpype")
    assert acpype[acpype.index("-n") + 1] == "1"
