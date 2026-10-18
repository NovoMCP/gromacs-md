"""Modified amino acids in the protein chain (PDB MODRES records).

Cleaning keeps only the 20 standard residues, so a modified residue such as
a phosphothreonine would be dropped and pdb2gmx would bond its neighbours
across the hole. Selenomethionine is converted to methionine instead (the
usual crystallographic substitution); anything else is refused at intake.
"""

from typing import List, NamedTuple

# Modified residue -> standard residue it is converted to before cleaning
CONVERTIBLE = {"MSE": "MET"}


class ModifiedResidue(NamedTuple):
    resname: str
    chain: str
    resnum: str
    parent: str
    description: str

    def label(self) -> str:
        desc = f" {self.description.lower()}" if self.description else ""
        return f"{self.resname} {self.chain}{self.resnum}{desc}"


def find_modified_residues(pdb_content: str) -> List[ModifiedResidue]:
    """MODRES records, one per residue, in file order."""
    found, seen = [], set()
    for line in pdb_content.splitlines():
        if not line.startswith("MODRES"):
            continue
        res = ModifiedResidue(
            resname=line[12:15].strip(),
            chain=line[16:17].strip(),
            resnum=line[18:23].strip(),
            parent=line[24:27].strip(),
            description=line[29:].strip(),
        )
        key = (res.resname, res.chain, res.resnum)
        if res.resname and key not in seen:
            seen.add(key)
            found.append(res)
    return found


def unsupported_modified_residues(pdb_content: str) -> List[ModifiedResidue]:
    """Modified residues that would be dropped from the chain (not convertible)."""
    return [r for r in find_modified_residues(pdb_content) if r.resname not in CONVERTIBLE]
