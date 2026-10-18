"""Ligand starting pose for protein-ligand MD.

An RDKit conformer built from SMILES sits around the origin, not in the
protein's frame, so pasting it next to a PDB-frame protein leaves the ligand
floating in solvent (for 181L, 18 Å from the nearest protein atom and ~27 Å
from the benzene pocket). The pose comes from docking instead: the autodock-gpu
service docks the ligand — boxed on the co-crystallized ligand when the PDB has
one, else on its auto-detected pocket — and the full-hydrogen ligand is rebuilt
on the top pose's heavy atoms.

Fallbacks, each reported in the job result's `ligand_placement`:
  docked                 — top docking pose (the normal path)
  native_site_centroid   — no docking: conformer centred on the co-crystallized
                           ligand's site, orientation not optimized
  unplaced               — neither: the legacy origin placement, flagged as not
                           a protein-ligand complex
"""

import json
import logging
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from .config import AUTODOCK_GPU_URL, AUTODOCK_GPU_API_KEY

logger = logging.getLogger(__name__)

PLACEMENT_FILE = "_ligand_placement.json"

# Waters, ions, buffers and crystallization additives: never the site ligand.
# Mirrors autodock-gpu's list, plus HED (2-hydroxyethyl disulfide), the
# additive in most T4 lysozyme structures — in 181L it outnumbers the benzene.
_NON_LIGAND_HETATMS = {
    "HOH", "DOD", "WAT", "H2O",
    "NA", "CL", "K", "MG", "CA", "ZN", "FE", "MN", "CU", "NI", "CO", "CD",
    "HG", "AU", "AG", "LI", "CS", "BR", "IOD", "F", "BA", "SR",
    "PO4", "SO4", "NO3", "NH4", "ACT", "FMT", "EDO", "PEG", "PGE", "PG4",
    "GOL", "MPD", "MES", "HEPES", "TRIS", "BIS", "BCT", "CIT", "TLA",
    "LDA", "LMT", "BOG", "OGA", "DDR", "C8E", "CYC",
    "GLC", "SUC", "TRE", "MAL", "LAT",
    "DTT", "BME", "MRC", "HED",
    # lipids, sterols and detergents from membrane-protein crystals: large, so
    # "largest HETATM" picked 2RH1's cholesterol site over carazolol's pocket
    "CLR", "CHS", "Y01", "CHD", "PLM", "MYR", "OLA", "OLB", "OLC", "PEE", "POV", "PCW", "LFA",
    "D10", "D12", "UND", "DMU", "LMU", "UMQ", "BNG", "HTG", "SOG", "12P", "1PE", "P6G", "BU1",
}

_DOCK_BOX_A = 22.0
_DOCK_TIMEOUT_S = 600.0


def _heavy_atoms(smiles: Optional[str]) -> Optional[int]:
    if not smiles:
        return None
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    return mol.GetNumHeavyAtoms() if mol is not None else None


def native_ligand_site(pdb_content: str, ligand_smiles: Optional[str] = None) -> Optional[dict]:
    """The co-crystallized ligand's site: the non-additive HETATM residue
    closest in heavy-atom count to the submitted ligand (the largest one
    without a SMILES). Call before HETATM records are stripped."""
    residues: dict = {}
    for line in pdb_content.splitlines():
        if not line.startswith("HETATM") or len(line) < 54:
            continue
        resname = line[17:20].strip()
        if resname in _NON_LIGAND_HETATMS:
            continue
        element = (line[76:78].strip() if len(line) >= 78 else "") or line[12:16].strip()[:1]
        if element.upper() == "H":
            continue
        key = (line[21], line[22:27].strip(), resname)
        residues.setdefault(key, []).append(
            (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        )
    if not residues:
        return None
    n_ligand = _heavy_atoms(ligand_smiles)
    if n_ligand:
        # closest in size, the larger on a tie
        (chain, resseq, resname), xyz = min(
            residues.items(), key=lambda kv: (abs(len(kv[1]) - n_ligand), -len(kv[1])))
    else:
        (chain, resseq, resname), xyz = max(residues.items(), key=lambda kv: len(kv[1]))
    center = np.mean(np.array(xyz), axis=0)
    return {
        "resname": resname,
        "chain": chain.strip(),
        "resseq": resseq,
        "n_heavy_atoms": len(xyz),
        "center": [round(float(v), 3) for v in center],
    }


def dock_top_pose(ligand_smiles: str, protein_pdb: str, site: Optional[dict]) -> dict:
    """Dock with autodock-gpu and return the best-scoring pose (with `pose_pdb`)."""
    payload = {
        "ligand_smiles": ligand_smiles,
        "protein_pdb_content": protein_pdb,
        "num_poses": 3,
        "use_addie_reranking": False,
        "enable_reference_docking": False,
        "return_pose_pdb": True,
    }
    if site:
        cx, cy, cz = site["center"]
        payload.update(
            center_x=cx, center_y=cy, center_z=cz, auto_detect_binding_site=False,
            size_x=_DOCK_BOX_A, size_y=_DOCK_BOX_A, size_z=_DOCK_BOX_A,
        )
    r = httpx.post(
        f"{AUTODOCK_GPU_URL.rstrip('/')}/dock",
        json=payload,
        headers={"X-API-Key": AUTODOCK_GPU_API_KEY},
        timeout=_DOCK_TIMEOUT_S,
    )
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "completed":
        raise RuntimeError(f"docking {data.get('status')}: {data.get('error')}")
    poses = [p for p in data.get("poses") or [] if p.get("pose_pdb")]
    if not poses:
        raise RuntimeError(
            "docking returned no pose structures (autodock-gpu without return_pose_pdb support?)"
        )
    return min(poses, key=lambda p: p["score"])


def ligand_from_pose(pose_pdb: str, ligand_smiles: str, out_pdb: Path) -> None:
    """Write the full-hydrogen ligand on the docked heavy-atom coordinates.

    The pose holds heavy atoms and polar H in the docking program's atom order.
    Bond orders and formal charges come from the SMILES; every H is rebuilt and
    relaxed with the heavy atoms held fixed, so the docked pose is kept exactly.
    """
    pose = Chem.MolFromPDBBlock(pose_pdb, removeHs=False, sanitize=False, proximityBonding=True)
    if pose is None:
        raise ValueError("could not parse the docked pose")
    rw = Chem.RWMol(pose)
    for idx in sorted((a.GetIdx() for a in rw.GetAtoms() if a.GetAtomicNum() == 1), reverse=True):
        rw.RemoveAtom(idx)
    template = Chem.MolFromSmiles(ligand_smiles)
    if template is None:
        raise ValueError(f"Invalid SMILES: {ligand_smiles}")
    mol = AllChem.AssignBondOrdersFromTemplate(template, rw.GetMol())
    # AssignBondOrdersFromTemplate keeps bond orders; carry the template's
    # formal charges over too (charged ligands would otherwise come out neutral).
    match = mol.GetSubstructMatch(Chem.RemoveHs(template))
    if len(match) != template.GetNumHeavyAtoms():
        raise ValueError("docked pose does not match the ligand SMILES")
    for t_idx, m_idx in enumerate(match):
        mol.GetAtomWithIdx(m_idx).SetFormalCharge(template.GetAtomWithIdx(t_idx).GetFormalCharge())
    mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(mol)

    mol_h = Chem.AddHs(mol, addCoords=True)
    ff = AllChem.UFFGetMoleculeForceField(mol_h)
    if ff is not None:
        for atom in mol_h.GetAtoms():
            if atom.GetAtomicNum() > 1:
                ff.AddFixedPoint(atom.GetIdx())
        ff.Minimize(maxIts=500)
    # One residue for the whole ligand, as smiles_to_pdb writes it: ACPYPE
    # refuses a ligand split over residues, and the pose's LIG/L labels sit on
    # the heavy atoms only. A mol-block round trip drops the PDB residue info
    # and keeps coordinates and charges.
    mol_h = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol_h), removeHs=False)
    Chem.MolToPDBFile(mol_h, str(out_pdb))


def _conformer_at(ligand_smiles: str, out_pdb: Path, center=None) -> None:
    """RDKit conformer (as smiles_to_pdb builds it), optionally centred on `center`."""
    mol = Chem.MolFromSmiles(ligand_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {ligand_smiles}")
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    if center is not None:
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        shift = np.array(center) - pos.mean(axis=0)
        for i, p in enumerate(pos + shift):
            conf.SetAtomPosition(i, p.tolist())
    Chem.MolToPDBFile(mol, str(out_pdb))


def place_ligand(ligand_smiles: str, protein_pdb: str, site: Optional[dict], out_pdb: Path) -> dict:
    """Write the ligand's starting structure to out_pdb; return how it was placed."""
    if AUTODOCK_GPU_URL:
        try:
            pose = dock_top_pose(ligand_smiles, protein_pdb, site)
            ligand_from_pose(pose["pose_pdb"], ligand_smiles, out_pdb)
            box = (f"co-crystallized {site['resname']} site" if site
                   else "autodock-gpu's detected pocket")
            logger.info(f"Ligand docked for MD start: {pose['score']} kcal/mol ({box})")
            return {
                "method": "docked",
                "docking_score_kcal_mol": pose["score"],
                "docking_box": box,
                "native_ligand": site,
            }
        except Exception as e:  # noqa: BLE001 — placement falls back, never fails the job
            reason = f"docking failed: {e}"
            logger.warning(f"Ligand placement: {reason}")
    else:
        reason = "AUTODOCK_GPU_URL not set"

    if site:
        _conformer_at(ligand_smiles, out_pdb, center=site["center"])
        return {
            "method": "native_site_centroid",
            "native_ligand": site,
            "warning": (
                f"No docked pose ({reason}). The ligand conformer is centred on the "
                f"co-crystallized {site['resname']} site; its orientation is not optimized."
            ),
        }

    _conformer_at(ligand_smiles, out_pdb)
    return {
        "method": "unplaced",
        "warning": (
            f"No docked pose ({reason}) and no co-crystallized ligand to locate the "
            "binding site. The ligand starts at its RDKit coordinates, outside the "
            "protein — results describe a protein and a free ligand, not a complex."
        ),
    }


def save_placement(workspace: Path, placement: dict) -> None:
    (Path(workspace) / PLACEMENT_FILE).write_text(json.dumps(placement))


def load_placement(workspace: Path) -> Optional[dict]:
    path = Path(workspace) / PLACEMENT_FILE
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except ValueError:
        return None
