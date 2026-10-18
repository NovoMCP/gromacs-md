"""Nucleotide + Mg2+ cofactors for the tleap build.

The nucleotide (GDP, GTP, ADP, ...) is built from its crystal heavy atoms
and its CCD entry's bonds, with every phosphate oxygen that carries a
hydrogen in the CCD deprotonated (GDP3-, GTP4-: their state at pH 7), and
parameterized with GAFF2 and AM1-BCC charges. Mg2+ is a nonbonded ion with
the Li/Merz 12-6 TIP3P parameters leaprc.water.tip3p loads, and the crystal
waters in its shell are kept so it starts fully coordinated.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from intake.cofactors import kept_cofactors

from .amber_build import AmberBuildError, LigandParams
from .covalent_build import _run
from .covalent_ligand import CovalentLigandError, parse_ccd

logger = logging.getLogger(__name__)

FORCE_FIELD_NOTE = "nucleotide GAFF2 (AM1-BCC), Mg2+ Li/Merz 12-6"

Xyz = Tuple[float, float, float]


@dataclass
class Cofactors:
    """What build_system adds to the protein: each nucleotide's parameters,
    and the Mg2+ ions with their shell waters as a PDB for tleap."""
    nucleotides: List[LigandParams]
    resnames: List[str]          # residue names that count as receptor in analysis
    ions_pdb_text: str
    info: Dict = field(default_factory=dict)


def cofactor_site(pdb_content: str) -> Optional[Dict]:
    """The cofactors MD keeps, with their crystal coordinates, read before
    cleaning strips HETATMs; None when there are none. First altloc only."""
    kept = kept_cofactors(pdb_content)
    if not kept["magnesium"]:
        return None
    nucleotides = {key: {} for key in kept["nucleotides"]}
    waters = {key for site in kept["magnesium"] for key in site.waters}
    water_xyz: Dict[Tuple[str, str, str], Xyz] = {}
    for line in pdb_content.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or line[16] not in " A":
            continue
        key = (line[17:20].strip(), line[21:22].strip(), line[22:27].strip())
        name = line[12:16].strip()
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if key in nucleotides:
            nucleotides[key].setdefault(name, xyz)
        elif key in waters and name == "O":
            water_xyz[key] = xyz
    return {
        "nucleotides": [{"code": res, "chain": chain, "resseq": seq, "atoms": atoms}
                        for (res, chain, seq), atoms in sorted(nucleotides.items())],
        "magnesium": [{"chain": s.chain, "resseq": s.resseq, "xyz": s.xyz,
                       "nucleotide": f"{s.nucleotide[0]} {s.nucleotide[1]}{s.nucleotide[2]}",
                       "waters": len(s.waters)} for s in kept["magnesium"]],
        "waters": [water_xyz[k] for k in sorted(water_xyz)],
    }


def nucleotide_mol(cif_text: str, crystal: Dict[str, Xyz]):
    """RDKit Mol of the nucleotide at pH 7 on its crystal heavy atoms,
    hydrogens added with coordinates."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    atoms, bonds = parse_ccd(cif_text)
    element = {a.name: a.element for a in atoms}
    heavy = [a for a in atoms if a.element != "H"]
    missing = [a.name for a in heavy if a.name not in crystal]
    if missing:
        raise CovalentLigandError(f"atoms missing from the crystal nucleotide: {', '.join(missing[:8])}")
    # phosphate oxygens that carry a hydrogen in the CCD lose it at pH 7
    bonded: Dict[str, set] = {}
    for b in bonds:
        bonded.setdefault(b.a, set()).add(element.get(b.b))
        bonded.setdefault(b.b, set()).add(element.get(b.a))
    acidic = {n for n, e in element.items() if e == "O" and {"P", "H"} <= bonded.get(n, set())}

    mol = Chem.RWMol()
    idx = {}
    for a in heavy:
        at = Chem.Atom(a.element)
        at.SetFormalCharge(-1 if a.name in acidic else a.charge)
        idx[a.name] = mol.AddAtom(at)
    order = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE,
             "TRIP": Chem.BondType.TRIPLE, "AROM": Chem.BondType.AROMATIC}
    for b in bonds:
        if b.a in idx and b.b in idx:
            mol.AddBond(idx[b.a], idx[b.b], order[b.order])
    conf = Chem.Conformer(mol.GetNumAtoms())
    for a in heavy:
        conf.SetAtomPosition(idx[a.name], Point3D(*crystal[a.name]))
    mol.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise CovalentLigandError(f"nucleotide isn't a valid molecule: {e}")
    return Chem.AddHs(mol.GetMol(), addCoords=True)


def parameterize_nucleotide(workspace: Path, code: str, cif_text: str,
                            crystal: Dict[str, Xyz]) -> Tuple[LigandParams, int]:
    from rdkit import Chem

    mol = nucleotide_mol(cif_text, crystal)
    charge = Chem.GetFormalCharge(mol)
    d = workspace / f"cofactor_{code.lower()}"
    d.mkdir(exist_ok=True)
    Chem.MolToMolFile(mol, str(d / f"{code}.mol"))
    logger.info(f"cofactors: AM1-BCC charges for {code} ({mol.GetNumAtoms()} atoms, net charge {charge})")
    _run(["antechamber", "-i", f"{code}.mol", "-fi", "mdl", "-o", f"{code}.mol2", "-fo", "mol2",
          "-c", "bcc", "-at", "gaff2", "-nc", str(charge), "-rn", code, "-pf", "y"],
         d, f"antechamber ({code}, AM1-BCC)")
    _run(["parmchk2", "-i", f"{code}.mol2", "-f", "mol2", "-o", f"{code}.frcmod", "-s", "2"],
         d, f"parmchk2 ({code})")
    return LigandParams(d / f"{code}.mol2", d / f"{code}.frcmod"), charge


def ions_pdb(site: Dict) -> str:
    """Mg2+ ions (tleap's MG) and their shell waters (WAT, oxygen only; tleap
    adds the hydrogens), one residue each."""
    lines, serial = [], 0
    for mg in site["magnesium"]:
        serial += 1
        x, y, z = mg["xyz"]
        lines.append(f"HETATM{serial:5d} MG    MG A{serial:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          MG\nTER\n")
    for x, y, z in site["waters"]:
        serial += 1
        lines.append(f"HETATM{serial:5d}  O   WAT W{serial:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           O\nTER\n")
    return "".join(lines) + "END\n"


def prepare_cofactors(workspace: Path, site: Dict, ccds: Dict[str, str]) -> Cofactors:
    """Parameters for every kept nucleotide (one set per residue name) and
    the ions PDB. `ccds` maps nucleotide code -> CCD mmCIF text."""
    params, charges = [], {}
    if len({n["code"] for n in site["nucleotides"]}) != len(site["nucleotides"]):
        # one mol2 per residue name: tleap would place copies at one pose
        raise AmberBuildError("more than one copy of a nucleotide; use one copy of the complex")
    for n in site["nucleotides"]:
        p, q = parameterize_nucleotide(workspace, n["code"], ccds[n["code"]], n["atoms"])
        params.append(p)
        charges[n["code"]] = q
    info = {
        "nucleotides": [{"residue": f"{n['code']} {n['chain']}{n['resseq']}", "net_charge": charges[n["code"]]}
                        for n in site["nucleotides"]],
        "magnesium": [{"residue": f"MG {m['chain']}{m['resseq']}", "with": m["nucleotide"],
                       "crystal_waters": m["waters"]} for m in site["magnesium"]],
        "parameters": FORCE_FIELD_NOTE,
    }
    return Cofactors(params, sorted(charges) + ["MG"], ions_pdb(site), info)
