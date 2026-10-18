"""Membrane systems: OPM orientation, packmol-memgen packing, tleap with Lipid21.

1. Orientation: OPM's file for the PDB ID has the membrane normal along z and
   the bilayer centre at z = 0. The submitted structure is superposed onto it
   on matching Cα atoms, so the crystal ligand (which locates the binding site
   for docking) moves into the same frame.
2. Packing: packmol-memgen places a POPC bilayer, water and 0.15 M NaCl
   around the protein and ligand. It only packs; parameters come from step 3.
3. Build: the packed system is parameterized by the same tleap path as the
   soluble branch (ff14SB + GAFF2 + TIP3P, plus Lipid21), converted to
   GROMACS with lipid head-group z restraints, and checked for energy parity.
"""

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import amber_build as ab

logger = logging.getLogger(__name__)

OPM_PDB_URL = "https://opm-assets.storage.googleapis.com/pdb/{pdb_id}.pdb"
LIPID = "POPC"
# packmol-memgen: lateral margin around the protein and water layer beyond it
# (Å). nloop_all 20 instead of its 100: minimization removes what's left, and
# packing a GPCR-sized patch otherwise takes hours on one core.
PACK_DIST_A = 15.0
PACK_WATER_A = 17.5
PACK_NLOOP_ALL = 20
FORCE_FIELD = "AMBER ff14SB + Lipid21 (POPC) + GAFF2 (AM1-BCC) + TIP3P, built with tleap"


class MembraneBuildError(ab.AmberBuildError):
    """The membrane system couldn't be oriented, packed or built."""


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------
def fetch_opm_oriented(pdb_id: str, timeout: float = 30.0) -> str:
    import httpx

    r = httpx.get(OPM_PDB_URL.format(pdb_id=pdb_id.lower()), timeout=timeout, follow_redirects=True)
    if r.status_code != 200:
        raise MembraneBuildError(f"OPM has no oriented structure for {pdb_id} (HTTP {r.status_code})")
    return r.text


def _ca_coordinates(pdb_text: str) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    out = {}
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[16] in " A":
            out.setdefault((line[21], line[22:27]),
                           (float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return out


def superpose_onto(pdb_text: str, reference_text: str, min_atoms: int = 20) -> Tuple[str, float]:
    """`pdb_text` with every ATOM/HETATM moved onto the reference frame by
    the Kabsch fit of matching Cα atoms (same chain and residue, or same
    residue if the reference lacks chain IDs). Returns (pdb text, rmsd Å)."""
    import numpy as np

    a, b = _ca_coordinates(pdb_text), _ca_coordinates(reference_text)
    keys = sorted(set(a) & set(b))
    if len(keys) < min_atoms:
        a = {(" ", k[1]): v for k, v in a.items()}
        b = {(" ", k[1]): v for k, v in b.items()}
        keys = sorted(set(a) & set(b))
    if len(keys) < min_atoms:
        raise MembraneBuildError(f"only {len(keys)} Cα atoms match the OPM structure")
    P = np.array([a[k] for k in keys])
    Q = np.array([b[k] for k in keys])
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    rmsd = float(np.sqrt((((P - pc) @ R.T + qc - Q) ** 2).sum(1).mean()))
    if rmsd > 2.0:
        raise MembraneBuildError(f"structure doesn't match OPM's ({rmsd:.1f} Å Cα RMSD)")
    out = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")) and len(line) >= 54:
            x = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            y = (x - pc) @ R.T + qc
            line = f"{line[:30]}{y[0]:8.3f}{y[1]:8.3f}{y[2]:8.3f}{line[54:]}"
        out.append(line)
    return "\n".join(out) + "\n", rmsd


def opm_half_thickness(opm_text: str) -> Optional[float]:
    m = re.search(r"1/2 of bilayer thickness:\s*([\d.]+)", opm_text)
    return float(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------
def _mol2_atoms(mol2: Path) -> Tuple[str, List[Tuple[str, str, Tuple[float, float, float]]]]:
    """(residue name, [(atom name, element, xyz)]) of a one-residue mol2."""
    atoms, resname, section = [], None, None
    for line in mol2.read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            continue
        if section == "@<TRIPOS>ATOM" and line.strip():
            f = line.split()
            # from the GAFF2 type (c3, cl, br, ...): an atom named CL1 may be carbon or chlorine
            element = {"cl": "Cl", "br": "Br"}.get(f[5][:2].lower(), f[5][0].upper())
            atoms.append((f[1], element, (float(f[2]), float(f[3]), float(f[4]))))
            resname = f[7]
    return resname, atoms


def complex_pdb(protein: ab.PreparedProtein, ligand: Optional[ab.LigandParams]) -> str:
    """Protein (tleap names) and the ligand, under the residue name of its
    mol2 so tleap matches it to the loaded unit."""
    lines = [l for l in protein.pdb_text.splitlines() if l.startswith(("ATOM", "TER"))]
    if ligand is not None:
        resname, atoms = _mol2_atoms(ligand.mol2)
        n = sum(1 for l in lines if l.startswith("ATOM"))
        for i, (name, element, xyz) in enumerate(atoms, start=1):
            padded = name if len(name) >= 4 else f" {name:<3}"
            lines.append(f"HETATM{(n + i) % 100000:5d} {padded:4s} {resname:>3s} X 999    "
                         f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}")
        lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


def pack(workspace: Path, complex_text: str, timeout: int = 7200) -> Tuple[Path, Dict]:
    """POPC bilayer, water and 0.15 M NaCl around the oriented complex."""
    d = workspace / "pack"
    d.mkdir(exist_ok=True)
    (d / "complex.pdb").write_text(complex_text)
    cmd = ["packmol-memgen", "--pdb", "complex.pdb", "--lipids", LIPID, "--ratio", "1",
           "--preoriented", "--notprotonate", "--nottrim", "--keepligs",
           "--salt", "--salt_c", "Na+", "--saltcon", str(ab.ION_CONC_M),
           "--dist", str(PACK_DIST_A), "--dist_wat", str(PACK_WATER_A),
           "--nloop_all", str(PACK_NLOOP_ALL), "--noprogress", "--overwrite", "--output", "packed.pdb"]
    proc = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout)
    packed = d / "packed.pdb"
    if proc.returncode != 0 or not packed.exists():
        raise MembraneBuildError(f"packmol-memgen failed: {((proc.stdout or '') + (proc.stderr or ''))[-600:]}")
    counts: Dict[str, int] = {}
    for line in packed.read_text().splitlines():
        name, res = line[12:16].strip(), line[17:21].strip()
        if line.startswith(("ATOM", "HETATM")) and (name in ("O", "Na+", "Cl-") or (res == "PC" and name.startswith("P"))):
            counts[res] = counts.get(res, 0) + 1
    stats = {"lipids": counts.get("PC", 0), "waters": counts.get("WAT", 0),
             "na": counts.get("Na+", 0), "cl": counts.get("Cl-", 0)}
    return packed, stats


def packed_box(packmol_inp: str, tolerance: float = 2.0) -> Tuple[float, float, float]:
    """Box edges (Å) of a packmol-memgen system: the union of the regions
    packmol filled (its `inside box` lines, written without a CRYST1), plus
    packmol's tolerance, which it keeps between molecules only inside a region
    and would otherwise be missing across the periodic boundary."""
    lo, hi = [float("inf")] * 3, [float("-inf")] * 3
    for line in packmol_inp.splitlines():
        f = line.split()
        if len(f) == 8 and f[:2] == ["inside", "box"]:
            v = [float(x) for x in f[2:]]
            lo = [min(a, b) for a, b in zip(lo, v[:3])]
            hi = [max(a, b) for a, b in zip(hi, v[3:])]
    if lo[0] == float("inf"):
        raise MembraneBuildError("packmol input has no box regions")
    return tuple(h - l + tolerance for l, h in zip(lo, hi))


def _tleap_packed_script(packed: Path, disulfides: Sequence[Tuple[int, int]],
                         ligand: Optional[ab.LigandParams], box: Tuple[float, float, float],
                         out_prefix: str) -> str:
    s = ["source leaprc.protein.ff14SB", "source leaprc.water.tip3p", "source leaprc.gaff2",
         "source leaprc.lipid21"]
    if ligand is not None:
        resname, _ = _mol2_atoms(ligand.mol2)
        s.append(f"loadamberparams {ligand.frcmod.name}")
        s.append(f"{resname} = loadmol2 {ligand.mol2.name}")
    s.append(f"sys = loadpdb {packed.name}")
    s += [f"bond sys.{i}.SG sys.{j}.SG" for i, j in disulfides]
    s.append(f"set sys box {{ {box[0]:.3f} {box[1]:.3f} {box[2]:.3f} }}")
    s.append(f"saveamberparm sys {out_prefix}.prmtop {out_prefix}.inpcrd")
    s.append("quit")
    return "\n".join(s) + "\n"


def build_membrane_for_pipeline(workspace: Path, compound_id: str, protein_pdb_text: str,
                                ligand_pdb: Optional[Path], ligand_smiles: Optional[str],
                                orientation: Dict) -> Tuple[Path, Path, Dict]:
    """system_prep for a membrane protein (already in the OPM frame): pack,
    build with tleap, convert, check energy parity. Saves <cid>_build_info.json."""
    import shutil
    from .topology import generate_ligand_topology

    protein = ab.prepare_protein(protein_pdb_text)
    ligand = None
    if ligand_pdb is not None:
        generate_ligand_topology(workspace, compound_id, ligand_pdb, ligand_smiles)
        base = compound_id.replace("-", "_")
        acpype_dir = workspace / f"{base}.acpype"
        ligand = ab.LigandParams(acpype_dir / f"{base}_bcc_gaff2.mol2", acpype_dir / f"{base}_AC.frcmod")

    logger.info("membrane: packing POPC bilayer, water and ions (packmol-memgen)")
    packed, stats = pack(workspace, complex_pdb(protein, ligand))
    box = packed_box((packed.parent / "packmol.inp").read_text())

    shutil.copy(packed, workspace / "packed.pdb")
    if ligand is not None:
        for p in (ligand.mol2, ligand.frcmod):
            shutil.copy(p, workspace / p.name)
    prefix = f"{compound_id}_amber"
    ab._run_tleap(workspace, _tleap_packed_script(workspace / "packed.pdb", protein.disulfides,
                                                  ligand, box, prefix), "leap_build")
    prmtop, inpcrd = workspace / f"{prefix}.prmtop", workspace / f"{prefix}.inpcrd"
    gro, top, groups = ab.convert_to_gromacs(workspace, compound_id, prmtop, inpcrd, protein)
    parity = ab.check_energy_parity(workspace, compound_id, prmtop, inpcrd, gro, top)

    info = {
        "builder": "tleap",
        "force_field": FORCE_FIELD,
        "ions": {"na": stats["na"], "cl": stats["cl"]},
        "disulfides": len(protein.disulfides),
        "energy_parity": {t: v["diff"] for t, v in parity["terms"].items()},
        "notes": protein.notes,
        "membrane": {
            "lipid": LIPID,
            "lipids": stats["lipids"],
            "waters": stats["waters"],
            "box_angstrom": [round(v, 1) for v in box],
            "orientation": orientation,
        },
    }
    ab.build_info_path(workspace, compound_id).write_text(json.dumps(info))
    return gro, top, info
