"""The bonded form of a covalent ligand, from the crystal and the CCD.

A crystal structure gives a covalent ligand's heavy-atom positions and the
LINK to the protein atom (usually Cys SG), but no hydrogens or bond orders.
Those come from the ligand's RCSB Chemical Component Dictionary (CCD) entry,
which isn't consistently the bonded form:

  YY3 (osimertinib)  unreacted acrylamide, C8=C9
  8E8 (ibrutinib)    bonded form with H in place of S (a methyl)
  MOV (sotorasib)    "bound form", methyl
  4WI (nirmatrelvir) thioimidate drawn as an aldimine, H on C3

so the bond to the protein is formed by warhead rule (`form_bond`), and the
result is the ligand with its bonding atom capped as in the protein: S-CH3 at
the SG and CB positions for a cysteine adduct. Hydrogens follow from valence.
"""

import logging
import shlex
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

CCD_URL = "https://files.rcsb.org/ligands/download/{code}.cif"

_ORDER = {"SING": 1, "DOUB": 2, "TRIP": 3, "AROM": 1.5}
HALOGENS = {"Cl", "Br", "I"}


class CovalentLigandError(ValueError):
    """The ligand's bonded form can't be built from the crystal and its CCD entry."""


@dataclass
class CcdAtom:
    name: str
    element: str
    charge: int


@dataclass
class CcdBond:
    a: str
    b: str
    order: str


def _cif_loop(lines: List[str], category: str) -> List[Dict[str, str]]:
    """Rows of one `loop_` category of an mmCIF file, keyed by item name."""
    rows: List[Dict[str, str]] = []
    keys: List[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() == "loop_" and i + 1 < len(lines) and lines[i + 1].startswith(category + "."):
            i += 1
            while i < len(lines) and lines[i].startswith(category + "."):
                keys.append(lines[i].split(".", 1)[1].strip())
                i += 1
            values: List[str] = []
            while i < len(lines) and not lines[i].startswith(("#", "loop_", "_")):
                values += shlex.split(lines[i], posix=True)
                while len(values) >= len(keys):
                    rows.append(dict(zip(keys, values[:len(keys)])))
                    values = values[len(keys):]
                i += 1
            return rows
        i += 1
    return rows


def parse_ccd(cif_text: str) -> Tuple[List[CcdAtom], List[CcdBond]]:
    lines = cif_text.splitlines()
    atoms = [CcdAtom(r["atom_id"], r["type_symbol"].capitalize(), int(r.get("charge", "0") or 0))
             for r in _cif_loop(lines, "_chem_comp_atom")]
    bonds = [CcdBond(r["atom_id_1"], r["atom_id_2"], r["value_order"])
             for r in _cif_loop(lines, "_chem_comp_bond")]
    if not atoms:
        raise CovalentLigandError("no atoms in the CCD entry")
    return atoms, bonds


def form_bond(mol, x: int) -> str:
    """Make room on atom `x` for the bond to the protein, by warhead rule.
    Edits `mol` (an RWMol of heavy atoms, CCD bond orders); returns the rule."""
    from rdkit import Chem

    atom = mol.GetAtomWithIdx(x)
    bonds = list(atom.GetBonds())

    def other(b):
        return b.GetOtherAtom(atom)

    # Michael acceptor (acrylamide, vinyl sulfonamide, enone): x is the
    # beta carbon of C=C-C(=O) or C=C-S(=O); the C=C becomes single
    for b in bonds:
        y = other(b)
        if b.GetBondType() == Chem.BondType.DOUBLE and y.GetSymbol() == "C" and not b.GetIsAromatic():
            for b2 in y.GetBonds():
                z = b2.GetOtherAtom(y)
                if z.GetIdx() != x and z.GetSymbol() in ("C", "S") and any(
                        b3.GetBondType() == Chem.BondType.DOUBLE and b3.GetOtherAtom(z).GetSymbol() == "O"
                        for b3 in z.GetBonds()):
                    b.SetBondType(Chem.BondType.SINGLE)
                    return "michael_addition"
    for b in bonds:
        y = other(b)
        if b.GetBondType() == Chem.BondType.TRIPLE and y.GetSymbol() == "N":
            b.SetBondType(Chem.BondType.DOUBLE)
            return "nitrile_to_thioimidate"
    for b in bonds:
        y = other(b)
        if b.GetBondType() == Chem.BondType.DOUBLE and y.GetSymbol() == "O" and not any(
                other(b2).GetSymbol() in ("N", "O") for b2 in bonds if b2 is not b):
            b.SetBondType(Chem.BondType.SINGLE)
            return "carbonyl_to_hemithioacetal"
    for b in bonds:
        y = other(b)
        if y.GetSymbol() in HALOGENS and y.GetDegree() == 1:
            mol.RemoveAtom(y.GetIdx())
            return "halide_displaced"
    return "bonded_form"  # the CCD drew H where the protein atom goes


@dataclass
class ProteinSide:
    """Protein atoms attached to the ligand: (name, element, xyz) each, their
    bonds (name, name, order), and the atom the ligand bonds to."""
    atoms: List[Tuple[str, str, Tuple[float, float, float]]]
    bonds: List[Tuple[str, str, int]]
    attach: str


def methylthio_cap(sg_xyz, cb_xyz) -> ProteinSide:
    """Cys side chain reduced to S-CH3 (SG, CB)."""
    return ProteinSide([("SG", "S", sg_xyz), ("CB", "C", cb_xyz)], [("SG", "CB", 1)], "SG")


def capped_cysteine(res: Dict[str, Tuple[float, float, float]], prev: Dict[str, Tuple[float, float, float]],
                    nxt: Dict[str, Tuple[float, float, float]]) -> ProteinSide:
    """ACE-CYS-NME around the bonded cysteine, from crystal coordinates of
    the cysteine (`res`) and the residues before (`prev`: C, O, CA) and after
    (`nxt`: N, CA). The caps' methyls sit on the neighbours' CA."""
    try:
        atoms = [(n, n[0], res[n]) for n in ("N", "CA", "C", "O", "CB", "SG")]
        atoms += [("C_ACE", "C", prev["C"]), ("O_ACE", "O", prev["O"]), ("CH3_ACE", "C", prev["CA"]),
                  ("N_NME", "N", nxt["N"]), ("CH3_NME", "C", nxt["CA"])]
    except KeyError as e:
        raise CovalentLigandError(f"atom {e} missing around the bonded cysteine")
    bonds = [("N", "CA", 1), ("CA", "C", 1), ("C", "O", 2), ("CA", "CB", 1), ("CB", "SG", 1),
             ("C_ACE", "N", 1), ("C_ACE", "O_ACE", 2), ("C_ACE", "CH3_ACE", 1),
             ("C", "N_NME", 1), ("N_NME", "CH3_NME", 1)]
    return ProteinSide(atoms, bonds, "SG")


def bonded_ligand(cif_text: str, crystal: Dict[str, Tuple[float, float, float]], bond_atom: str,
                  protein: ProteinSide):
    """The ligand bonded to `protein` atoms, hydrogens added with
    coordinates, as an RDKit Mol.

    `crystal` maps the ligand's PDB atom names (which are its CCD atom ids)
    to their coordinates (Å). Returns (mol, rule, names): atom i < len(names)
    is names[i], the ligand's heavy atoms followed by the protein-side atoms;
    hydrogens come after.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    atoms, bonds = parse_ccd(cif_text)
    heavy = [a for a in atoms if a.element != "H"]
    # a leaving group on the bonded atom is gone from the crystal, as it should be
    leaving = {n for n in _neighbours(bonds, bond_atom)
               if any(a.name == n and a.element in HALOGENS for a in heavy)}
    missing = [a.name for a in heavy if a.name not in crystal and a.name not in leaving]
    if missing:
        raise CovalentLigandError(f"atoms missing from the crystal ligand: {', '.join(missing[:8])}")
    if bond_atom not in {a.name for a in heavy}:
        raise CovalentLigandError(f"bonded atom {bond_atom} is not in the CCD entry")

    mol = Chem.RWMol()
    idx = {}
    for a in heavy:
        at = Chem.Atom(a.element)
        at.SetFormalCharge(a.charge)
        idx[a.name] = mol.AddAtom(at)
    for b in bonds:
        if b.a in idx and b.b in idx:
            order = {"SING": Chem.BondType.SINGLE, "DOUB": Chem.BondType.DOUBLE,
                     "TRIP": Chem.BondType.TRIPLE}.get(b.order)
            if order is None:
                raise CovalentLigandError(f"unsupported CCD bond order {b.order}")
            mol.AddBond(idx[b.a], idx[b.b], order)

    names = [a.name for a in heavy]
    rule = form_bond(mol, idx[bond_atom])
    if rule == "halide_displaced":  # an atom went; reindex names by what's left
        names = _names_after_removal(heavy, bonds, bond_atom)
    xyz = [crystal[n] for n in names]
    bond_idx = names.index(bond_atom)
    side = {}  # protein atoms by name: ligands reuse names such as N and C
    for name, element, pos in protein.atoms:
        side[name] = mol.AddAtom(Chem.Atom(element))
        names.append(name)
        xyz.append(pos)
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE}
    for a, b, o in protein.bonds:
        mol.AddBond(side[a], side[b], order[o])
    mol.AddBond(bond_idx, side[protein.attach], Chem.BondType.SINGLE)

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, pos in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(*pos))
    mol.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(mol)
    except Exception as e:
        raise CovalentLigandError(f"bonded form ({rule}) isn't a valid molecule: {e}")
    out = Chem.AddHs(mol.GetMol(), addCoords=True)
    return out, rule, names


def _names_after_removal(heavy: List[CcdAtom], bonds: List[CcdBond], bond_atom: str) -> List[str]:
    """Heavy-atom names once the halide on `bond_atom` is gone (RDKit keeps
    the order of the remaining atoms)."""
    neighbours = _neighbours(bonds, bond_atom)
    gone = next(a.name for a in heavy if a.name in neighbours and a.element in HALOGENS)
    return [a.name for a in heavy if a.name != gone]


def _neighbours(bonds: List[CcdBond], name: str) -> set:
    return {b.b for b in bonds if b.a == name} | {b.a for b in bonds if b.b == name}
