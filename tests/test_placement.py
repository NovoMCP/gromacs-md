"""Tests for the protein-ligand starting pose."""

import numpy as np
import pytest

try:
    from rdkit import Chem
    from gromacs_md import placement
except ImportError:
    pytest.skip("gromacs_md dependencies not installed", allow_module_level=True)


def _het(serial, name, resname, resseq, x, y, z, element):
    return (f"HETATM{serial:5d} {name:<4s} {resname:>3s} A{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}")


# 181L-like: a 6-atom benzene, an 8-atom HED additive, a chloride and a water.
SITE_PDB = "\n".join(
    [_het(i + 1, f"C{i + 1}", "BNZ", 166, 26.9 + np.cos(i) , 6.1 + np.sin(i), 4.2, "C") for i in range(6)]
    + [_het(10 + i, f"C{i + 1}", "HED", 170, 40.0 + i, 0.0, 0.0, "C") for i in range(8)]
    + [_het(20, "CL", " CL", 173, 1.0, 1.0, 1.0, "CL"), _het(21, "O", "HOH", 200, 0.0, 0.0, 0.0, "O")]
) + "\nEND\n"


def test_native_site_skips_additives_ions_and_water():
    site = placement.native_ligand_site(SITE_PDB)
    assert site["resname"] == "BNZ"
    assert site["n_heavy_atoms"] == 6
    assert np.allclose(site["center"], [26.9 + np.mean(np.cos(range(6))), 6.1 + np.mean(np.sin(range(6))), 4.2], atol=1e-3)


def test_no_native_site_without_a_ligand():
    assert placement.native_ligand_site(_het(1, "O", "HOH", 1, 0, 0, 0, "O")) is None


def _pose_pdb(smiles, offset):
    """A docked-pose stand-in: heavy atoms + polar H, shuffled, in a far frame."""
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    pos = mol.GetConformer().GetPositions() + offset
    keep = [a.GetIdx() for a in mol.GetAtoms()
            if a.GetAtomicNum() > 1 or a.GetNeighbors()[0].GetAtomicNum() in (7, 8)]
    keep = keep[::-1]
    lines = [_het(n + 1, mol.GetAtomWithIdx(i).GetSymbol(), "UNL", 1, *pos[i], mol.GetAtomWithIdx(i).GetSymbol())
             for n, i in enumerate(keep)]
    heavy = np.array([pos[i] for i in keep if mol.GetAtomWithIdx(i).GetAtomicNum() > 1])
    return "\n".join(lines), heavy


@pytest.mark.parametrize("smiles", ["c1ccccc1", "CC(=O)[O-]", "C[NH3+]", "Oc1ccncc1"])
def test_ligand_from_pose_keeps_heavy_atoms_and_adds_every_h(tmp_path, smiles):
    pose, heavy = _pose_pdb(smiles, offset=np.array([30.0, 5.0, 4.0]))
    out = tmp_path / "lig.pdb"
    placement.ligand_from_pose(pose, smiles, out)
    mol = Chem.MolFromPDBFile(str(out), removeHs=False)
    ref = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert mol.GetNumAtoms() == ref.GetNumAtoms()
    got = np.array([p for a, p in zip(mol.GetAtoms(), mol.GetConformer().GetPositions()) if a.GetAtomicNum() > 1])
    # every docked heavy atom is kept in place (as a set — the order may differ)
    assert np.allclose(np.sort(got, axis=0), np.sort(heavy, axis=0), atol=1e-3)
    assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == Chem.GetFormalCharge(ref)
    # ACPYPE refuses a ligand split over residues ("Only ONE Residue is allowed")
    residues = {(a.GetPDBResidueInfo().GetResidueName(), a.GetPDBResidueInfo().GetResidueNumber())
                for a in mol.GetAtoms()}
    assert len(residues) == 1


def test_placement_falls_back_to_the_native_site(tmp_path, monkeypatch):
    monkeypatch.setattr(placement, "AUTODOCK_GPU_URL", "")
    site = placement.native_ligand_site(SITE_PDB)
    out = tmp_path / "lig.pdb"
    info = placement.place_ligand("c1ccccc1", "", site, out)
    assert info["method"] == "native_site_centroid"
    pos = Chem.MolFromPDBFile(str(out), removeHs=False).GetConformer().GetPositions()
    assert np.allclose(pos.mean(axis=0), site["center"], atol=1e-2)


def test_placement_flags_an_unplaced_ligand(tmp_path, monkeypatch):
    monkeypatch.setattr(placement, "AUTODOCK_GPU_URL", "")
    info = placement.place_ligand("c1ccccc1", "", None, tmp_path / "lig.pdb")
    assert info["method"] == "unplaced" and "not a complex" in info["warning"]


def test_docking_failure_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(placement, "AUTODOCK_GPU_URL", "http://autodock:8022")

    def boom(*a, **k):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(placement, "dock_top_pose", boom)
    info = placement.place_ligand("c1ccccc1", "", placement.native_ligand_site(SITE_PDB), tmp_path / "lig.pdb")
    assert info["method"] == "native_site_centroid" and "connection refused" in info["warning"]


def test_site_is_the_crystal_ligand_not_a_larger_lipid():
    # 2RH1: cholesterol (27 heavy atoms here) beside carazolol (4): the docking
    # box must go on carazolol's site, not the larger membrane lipid's
    pdb = "\n".join([_het(i, "C", "CLR", 412, 30.0 + i, 0, 0, "C") for i in range(27)]
                    + [_het(100 + i, "C", "CAU", 500, 0, 0, i, "C") for i in range(4)])
    site = placement.native_ligand_site(pdb)
    assert site["resname"] == "CAU"


def test_site_prefers_the_het_closest_in_size_to_the_ligand():
    pdb = "\n".join([_het(i, "C", "AAA", 1, i, 0, 0, "C") for i in range(30)]
                    + [_het(100 + i, "C", "BBB", 2, 0, i, 0, "C") for i in range(6)])
    assert placement.native_ligand_site(pdb)["resname"] == "AAA"            # no SMILES: largest
    assert placement.native_ligand_site(pdb, "c1ccccc1")["resname"] == "BBB"  # benzene: 6 heavy atoms
