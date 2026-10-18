"""Nucleotide-bound Mg2+ sites that MD keeps as cofactors.

A GTPase or kinase with GDP/GTP/ADP/ATP and its Mg2+ loses its fold around
the nucleotide pocket if both are stripped, so these sites are simulated:
the nucleotide with GAFF2 (AM1-BCC) in its pH 7 charge state, Mg2+ as a
nonbonded 12-6 ion, and the crystal waters in its shell. That holds for a
Mg2+ whose shell is all oxygen and includes a nucleotide phosphate; other
metals still go through the classifier's refusal rules.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

NUCLEOTIDES = {"GDP", "GTP", "GNP", "GSP", "GCP", "ADP", "ATP", "ANP", "ACP"}
WATERS = {"HOH", "WAT", "DOD"}
SHELL_A = 2.8  # Mg-O is about 2.1 A

Xyz = Tuple[float, float, float]
ResidueKey = Tuple[str, str, str]  # resname, chain, resseq


@dataclass
class MagnesiumSite:
    chain: str
    resseq: str
    xyz: Xyz
    nucleotide: Optional[ResidueKey] = None
    waters: List[ResidueKey] = field(default_factory=list)
    donors: List[str] = field(default_factory=list)  # "RES:CHAINSEQ:ATOM"
    refusal: Optional[str] = None  # why the site can't be kept, or None

    @property
    def supported(self) -> bool:
        return self.refusal is None

    def label(self) -> str:
        return f"MG {self.chain}{self.resseq}"


def _atoms(pdb_content: str):
    """(record, resname, chain, resseq, name, element, xyz) per atom, first altloc only."""
    for line in pdb_content.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or len(line) < 54 or line[16] not in " A":
            continue
        name = line[12:16].strip()
        element = (line[76:78].strip() if len(line) >= 78 else "") or name[:1]
        yield (line[:6].strip(), line[17:20].strip(), line[21:22].strip(), line[22:27].strip(),
               name, element.upper(), (float(line[30:38]), float(line[38:46]), float(line[46:54])))


def magnesium_sites(pdb_content: str) -> List[MagnesiumSite]:
    """Every Mg2+ in the structure, with whether MD can keep it as a cofactor."""
    atoms = list(_atoms(pdb_content))
    sites = [MagnesiumSite(chain, seq, xyz) for rec, res, chain, seq, name, el, xyz in atoms
             if rec == "HETATM" and res == "MG" and el == "MG"]
    for site in sites:
        nucleotides = set()
        for rec, res, chain, seq, name, el, xyz in atoms:
            if (res, chain, seq) == ("MG", site.chain, site.resseq) or math.dist(xyz, site.xyz) > SHELL_A:
                continue
            if el == "H":
                continue
            site.donors.append(f"{res}:{chain}{seq}:{name}")
            if el != "O":
                site.refusal = f"{name} of {res} {chain}{seq} in the shell; only oxygen donors are supported"
            elif res in WATERS:
                site.waters.append((res, chain, seq))
            elif res in NUCLEOTIDES:
                nucleotides.add((res, chain, seq))
        if site.refusal is None:
            if not nucleotides:
                site.refusal = "no nucleotide phosphate in the shell"
            elif len(nucleotides) > 1:
                site.refusal = "more than one nucleotide in the shell"
            else:
                site.nucleotide = nucleotides.pop()
    return sites


def kept_cofactors(pdb_content: str) -> Dict[str, object]:
    """What MD keeps: the Mg2+ sites it supports and their nucleotides.
    Empty lists when there's nothing to keep."""
    sites = [s for s in magnesium_sites(pdb_content) if s.supported]
    nucleotides = sorted({s.nucleotide for s in sites if s.nucleotide})
    return {"magnesium": sites, "nucleotides": nucleotides}
