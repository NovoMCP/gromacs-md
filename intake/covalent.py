"""Covalent ligand detection from PDB LINK records.

A ligand bonded to a protein residue (e.g. the N3 inhibitor on Cys145 of
SARS-CoV-2 Mpro, 6LU7) can't be run by the soluble branch: protein
cleaning drops the HETATM part, keeps any standard-residue part of the
inhibitor as a stray peptide, and the docked SMILES is simulated as a
non-bonded ligand. Glycans, lipidation, caps, metals and waters also
appear in LINK records and are handled elsewhere, so they don't count.
"""

from __future__ import annotations

from typing import List

STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}
WATER = {"HOH", "WAT", "DOD", "H2O"}
# N- and O-glycan sugars: stripped with the other HETATMs, as before
SUGARS = {
    "NAG", "NDG", "MAN", "BMA", "FUC", "FUL", "GAL", "GLA", "GLC", "BGC",
    "SIA", "XYS", "A2G", "NGA", "GCS", "BDP",
}
# Lipidation, terminal caps and crystallization chemistry (alkylated Cys):
# modifications, not bound ligands; stripped like other HETATMs (2RH1 has
# a palmitoyl and an acetamide on cysteines)
MODIFICATIONS = {
    "PLM", "MYR", "FAR", "GER", "ACE", "NH2", "FOR", "ACM", "CMT", "BME", "CSO",
}
# Metal ions and halides; metals are classified by the metal-site branch
IONS = {
    "ZN", "CA", "MG", "MN", "FE", "FE2", "CU", "CU1", "CO", "NI", "CD", "HG",
    "NA", "K", "LI", "SR", "BA", "CS", "RB", "CL", "BR", "IOD", "F",
}


def _side(line: str, start: int) -> tuple[str, str, str, str]:
    """(atom, resname, chain, resseq) of one partner of a LINK record."""
    return (
        line[start:start + 4].strip(),
        line[start + 5:start + 8].strip(),
        line[start + 9:start + 10].strip(),
        line[start + 10:start + 14].strip(),
    )


def find_covalent_ligand_links(pdb_content: str) -> List[str]:
    """LINK records joining a ligand residue to a standard amino acid."""
    found: List[str] = []
    for line in pdb_content.splitlines():
        if not line.startswith("LINK"):
            continue
        a, b = _side(line, 12), _side(line, 42)
        for lig, res in ((a, b), (b, a)):
            name = lig[1]
            if res[1] in STANDARD_AA and name not in STANDARD_AA | WATER | SUGARS | MODIFICATIONS | IONS:
                dist = line[73:78].strip()
                found.append(
                    f"{name} {lig[2]}{lig[3]} {lig[0]} - {res[1]} {res[2]}{res[3]} {res[0]}"
                    + (f" ({dist} A)" if dist else "")
                )
                break
    return found
