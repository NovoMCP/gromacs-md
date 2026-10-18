"""Covalent ligand detection from PDB LINK records.

A ligand bonded to a protein residue (e.g. the N3 inhibitor on Cys145 of
SARS-CoV-2 Mpro, 6LU7) can't be run by the soluble branch: protein
cleaning drops the HETATM part, keeps any standard-residue part of the
inhibitor as a stray peptide, and the docked SMILES is simulated as a
non-bonded ligand. Glycans, lipidation, caps, metals and waters also
appear in LINK records and are handled elsewhere, so they don't count.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

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


class CovalentLink(NamedTuple):
    """One LINK between a ligand atom and a protein residue atom."""
    ligand: str
    ligand_chain: str
    ligand_resseq: str
    ligand_atom: str
    residue: str
    chain: str
    resseq: str
    atom: str
    distance: str = ""

    def label(self) -> str:
        return (f"{self.ligand} {self.ligand_chain}{self.ligand_resseq} {self.ligand_atom} - "
                f"{self.residue} {self.chain}{self.resseq} {self.atom}"
                + (f" ({self.distance} A)" if self.distance else ""))


def _not_ligand(name: str, in_chain: set) -> bool:
    return name in STANDARD_AA | WATER | SUGARS | MODIFICATIONS | IONS | in_chain


def covalent_ligand_link_records(pdb_content: str) -> List[CovalentLink]:
    """LINK records joining a ligand residue to a standard amino acid.

    Modified residues listed in MODRES (a phosphoserine, say) are part of the
    chain: their peptide-bond LINKs are not ligand links.
    """
    found: List[CovalentLink] = []
    in_chain = {line[12:15].strip() for line in pdb_content.splitlines() if line.startswith("MODRES")}
    for line in pdb_content.splitlines():
        if not line.startswith("LINK"):
            continue
        a, b = _side(line, 12), _side(line, 42)
        for lig, res in ((a, b), (b, a)):
            if res[1] in STANDARD_AA and not _not_ligand(lig[1], in_chain):
                found.append(CovalentLink(lig[1], lig[2], lig[3], lig[0], res[1], res[2], res[3], res[0],
                                          line[73:78].strip()))
                break
    return found


def find_covalent_ligand_links(pdb_content: str) -> List[str]:
    """LINK records joining a ligand residue to a standard amino acid, as text."""
    return [link.label() for link in covalent_ligand_link_records(pdb_content)]


def covalent_branch_refusal(pdb_content: str, links: List[CovalentLink]) -> Optional[str]:
    """Why the covalent branch can't run these links, or None if it can.

    It runs one ligand from a single HET residue, bonded to a cysteine SG
    (acrylamides, nitriles, chloroacetamides and the like). A ligand made of
    several HET residues bonded together (6LU7's peptidomimetic) and boron
    warheads (no GAFF2 types) are not supported yet.
    """
    if len(links) != 1:
        return (f"{len(links)} covalent ligand links; the covalent branch runs a structure "
                f"with one (use a single-chain structure, or one copy of the complex)")
    link = links[0]
    if (link.residue, link.atom) != ("CYS", "SG"):
        return f"ligand bonded to {link.residue} {link.atom}; the covalent branch supports cysteine SG only"
    in_chain = {line[12:15].strip() for line in pdb_content.splitlines() if line.startswith("MODRES")}
    for line in pdb_content.splitlines():
        if not line.startswith("LINK"):
            continue
        a, b = _side(line, 12), _side(line, 42)
        for one, other in ((a, b), (b, a)):
            if (one[1], one[2], one[3]) == (link.ligand, link.ligand_chain, link.ligand_resseq) \
                    and not _not_ligand(other[1], in_chain):
                return (f"the ligand is built from several residues ({link.ligand} bonded to "
                        f"{other[1]}); the covalent branch supports single-residue ligands")
    for line in pdb_content.splitlines():
        if line.startswith("HETATM") and line[17:20].strip() == link.ligand \
                and line[21:22].strip() == link.ligand_chain and line[22:26].strip() == link.ligand_resseq \
                and line[76:78].strip().upper() == "B":
            return "boron warhead; GAFF2 has no boron parameters"
    return None
