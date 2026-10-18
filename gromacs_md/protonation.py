"""The ligand's charge state at pH 7.

A SMILES usually arrives neutral, but most drugs are ions in water: an
aliphatic amine (pKa 9-11) is protonated, a carboxylic acid (pKa 3-5) is
not. Simulated neutral, the ligand loses the salt bridges that hold it
(carazolol's amine with Asp113 in the beta2 receptor), and its AM1-BCC
charges are wrong everywhere.

Only groups whose pKa is at least about 2 units from 7 change, so one
state dominates (>99%). Groups near 7 (imidazole, N-alkyl morpholine,
pyridine, aniline, the second nitrogen of a piperazine) stay as submitted.
A SMILES that already carries formal charges is taken as the intended
state and left alone.
"""
from __future__ import annotations

from typing import List, Tuple

from rdkit import Chem

# acids: the atom that loses its proton
_ACIDS = [
    ("carboxylic acid", "[CX3](=O)[OX2H1]", 2),
    ("sulfonic acid", "[SX4](=O)(=O)[OX2H1]", 3),
    ("phosphate/phosphonate", "[PX4](=O)[OX2H1]", 2),   # first proton only (pKa 1-2)
    ("tetrazole", "[nH1]1nnnc1", 0),
    ("tetrazole", "n1[nH1]nnc1", 1),
    ("acyl sulfonamide", "[CX3](=O)[NX3H1][SX4](=O)=O", 2),
]

# bases: the atom that gains a proton
_AMIDINE = ("[NX2;!a;!$(N-[O,N]);!$(N-C=O);!$(N-S(=O)=O);!$(N-C#N)]"
            "=[CX3;!a]([#6,#1,N])[NX3;!a;!$(N-C=O);!$(N-S(=O)=O);!$(N-[O,N])]")
_AMINE = "[NX3;H2,H1,H0;+0;!$(N-[!#6;!#1]);!$(N-[#6;!X4])]"  # every neighbour an sp3 carbon


def _amine_ok(mol, atom, dist, protonated, carboxylates=()) -> bool:
    """An aliphatic amine whose pKa stays well above 7."""
    n = atom.GetIdx()
    ring = mol.GetRingInfo()
    for other in protonated:  # a second amine 1-3 bonds away (piperazine, diamine): pKa about 5-6
        if dist[n][other] <= 3:
            return False
    for a in mol.GetAtoms():
        d = dist[n][a.GetIdx()]
        if a.GetSymbol() == "F" and d <= 3:  # beta-fluorines: pKa 5-7
            return False
        if a.GetSymbol() == "O" and d == 3 and any(  # morpholine: pKa 7-8.5
                n in r and a.GetIdx() in r for r in ring.AtomRings()):
            return False
        if a.GetSymbol() == "C" and d == 2 and any(  # alpha-nitrile (pKa about 5) or propargyl (8)
                b.GetBondType() == Chem.BondType.TRIPLE for b in a.GetBonds()):
            return False
        if a.GetSymbol() in ("C", "S") and d == 2 and a.GetIdx() not in carboxylates and any(
                # alpha-amino ester/amide: pKa about 8 (an amino acid's carboxylate doesn't count)
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(a).GetSymbol() == "O"
                for b in a.GetBonds()):
            return False
    return True


def protonate_ph7(smiles: str) -> Tuple[str, List[str]]:
    """(SMILES at pH 7, notes on what changed). Unchanged, with no notes,
    when nothing applies or the SMILES already carries charges."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or any(a.GetFormalCharge() for a in mol.GetAtoms()):
        return smiles, []
    rw = Chem.RWMol(mol)
    changed: List[str] = []

    carboxylates = set()
    for name, smarts, i in _ACIDS:
        done = set()
        for match in mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            anchor = match[0]
            if anchor in done:  # one proton per phosphorus / tetrazole
                continue
            atom = rw.GetAtomWithIdx(match[i])
            if atom.GetFormalCharge() or atom.GetTotalNumHs() == 0:
                continue
            done.add(anchor)
            if name == "carboxylic acid":
                carboxylates.add(anchor)
            atom.SetFormalCharge(-1)
            atom.SetNumExplicitHs(atom.GetTotalNumHs() - 1)
            atom.SetNoImplicit(True)
            changed.append(name)

    dist = Chem.GetDistanceMatrix(mol)
    protonated: List[int] = []
    for match in mol.GetSubstructMatches(Chem.MolFromSmarts(_AMIDINE)):
        if any(dist[match[0]][p] <= 4 for p in protonated):  # a biguanide takes one proton
            continue
        atom = rw.GetAtomWithIdx(match[0])
        atom.SetFormalCharge(1)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
        atom.SetNoImplicit(True)
        protonated.append(match[0])
        changed.append("amidine/guanidine")
    # tertiary before secondary before primary: the most basic of a pair wins
    amines = sorted((m[0] for m in mol.GetSubstructMatches(Chem.MolFromSmarts(_AMINE))),
                    key=lambda i: -mol.GetAtomWithIdx(i).GetDegree())
    for n in amines:
        if not _amine_ok(mol, mol.GetAtomWithIdx(n), dist, protonated, carboxylates):
            continue
        atom = rw.GetAtomWithIdx(n)
        atom.SetFormalCharge(1)
        atom.SetNumExplicitHs(atom.GetTotalNumHs() + 1)
        atom.SetNoImplicit(True)
        protonated.append(n)
        changed.append("aliphatic amine")

    if not changed:
        return smiles, []
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    charge = Chem.GetFormalCharge(out)
    counts = {c: changed.count(c) for c in dict.fromkeys(changed)}
    groups = ", ".join(f"{n} {c}" if n > 1 else c for c, n in counts.items())
    return Chem.MolToSmiles(out), [f"charged at pH 7 ({groups}): net charge {charge:+d}"]
