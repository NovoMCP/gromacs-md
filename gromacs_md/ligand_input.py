"""Check a ligand SMILES before it reaches RDKit embedding and ACPYPE.

ACPYPE/antechamber fail on salts and on elements GAFF2 has no types for,
with an exit status and a log that never names the cause. Catch those up
front so the job fails (or the request is rejected) with a reason.
"""

from typing import List, Tuple

from rdkit import Chem

# Elements GAFF2 has atom types for
GAFF2_ELEMENTS = {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}


class LigandInputError(ValueError):
    """The SMILES can't be parameterized; the message says why."""


def prepare_ligand_smiles(smiles: str) -> Tuple[str, List[str]]:
    """Return (SMILES to simulate, notes about what was changed).

    A multi-component SMILES (a salt or solvate, e.g. "imatinib mesylate")
    keeps its largest component; the counter-ions and solvent are dropped
    and listed in the notes.
    """
    mol = Chem.MolFromSmiles(smiles or "")
    if mol is None:
        raise LigandInputError(f"not a valid SMILES: {smiles!r}")

    notes: List[str] = []
    frags = sorted(Chem.GetMolFrags(mol, asMols=True), key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    if len(frags) > 1:
        if frags[0].GetNumHeavyAtoms() == frags[1].GetNumHeavyAtoms():
            raise LigandInputError(
                f"the SMILES holds several molecules of the same size "
                f"({', '.join(Chem.MolToSmiles(f) for f in frags)}); submit one"
            )
        dropped = ", ".join(Chem.MolToSmiles(f) for f in frags[1:])
        notes.append(f"kept the largest component; dropped {dropped}")
        mol = frags[0]

    unsupported = sorted({a.GetSymbol() for a in mol.GetAtoms()} - GAFF2_ELEMENTS)
    if unsupported:
        raise LigandInputError(
            f"the ligand contains {', '.join(unsupported)}, which GAFF2 has no parameters for "
            f"(supported: {', '.join(sorted(GAFF2_ELEMENTS))})"
        )
    return Chem.MolToSmiles(mol), notes
