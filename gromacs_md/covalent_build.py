"""A covalently bound ligand as a modified cysteine residue for tleap.

The cysteine and the ligand bonded to it become one residue, CVX:

- charges: AM1-BCC on the capped fragment ACE-CYS(adduct)-NME (antechamber).
  The backbone atoms (N, H, CA, HA, C, O) keep their ff14SB charges, and the
  rest of the BCC charges is shifted evenly so the residue's net charge is the
  ligand's formal charge, as for any ff14SB residue.
- types: ff14SB for the backbone atoms, GAFF2 from CB onward. GAFF2
  parameters the ligand needs come from parmchk2 on the fragment.
- junction terms between the two (CA-CB, N-CA-CB, ...): tleap lists what it
  can't find, and each is taken from GAFF2 with the backbone types read as
  their GAFF2 equivalents (CX as c3, N as n, ...).

Out of scope: ligands built from several HET residues (6LU7's peptidomimetic),
residues other than cysteine, and boron (GAFF2 has no boron types).
"""

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .amber_build import AmberBuildError
from .config import ACPYPE_OMP_THREADS
from .covalent_ligand import CovalentLigandError, bonded_ligand, capped_cysteine

logger = logging.getLogger(__name__)

RESNAME = "CVX"

# ff14SB CYX backbone (amino12.lib): types and charges kept on the adduct residue
BACKBONE = {"N": ("N", -0.4157), "H": ("H", 0.2719), "CA": ("CX", 0.0429),
            "HA": ("H1", 0.0766), "C": ("C", 0.5973), "O": ("O", -0.5679)}
CAP_ATOMS = {"C_ACE", "O_ACE", "CH3_ACE", "N_NME", "CH3_NME"}
# ff14SB types next to the junction, read as GAFF2 types for missing terms
GAFF2_EQUIVALENT = {"N": "n", "H": "hn", "CX": "c3", "H1": "h1", "C": "c", "O": "o",
                    "CT": "c3", "HC": "hc", "HP": "hx", "2C": "c3", "3C": "c3", "N3": "n4",
                    "H4": "h4", "H5": "h5", "HA": "ha", "OH": "oh", "HO": "ho", "S": "ss",
                    "SH": "sh", "HS": "hs", "CA": "ca", "CC": "cc", "NA": "na", "NB": "nb",
                    "CW": "cc", "CR": "cc", "CV": "cd", "N2": "n2", "O2": "o", "CO": "c"}
WILDCARD = "X"


@dataclass
class Adduct:
    """What the build needs: residue library and parameter files, the residue's
    atoms as PDB lines (with hydrogens), and what was done."""
    lib: Path
    frcmods: List[Path]
    pdb_atoms: List[Tuple[str, str, Tuple[float, float, float]]]  # (name, element, xyz Å), CVX order
    ligand_atoms: List[str]   # CVX atom names that belong to the ligand
    bond_atom: str            # CVX name of the ligand atom bonded to SG
    rule: str
    net_charge: int
    notes: List[str] = field(default_factory=list)


def _residue_names(mol, names: List[str]) -> Dict[int, str]:
    """CVX atom name per fragment atom index, caps excluded.

    Backbone and CB/SG keep their PDB names, hydrogens on them take the
    ff14SB names; ligand heavy atoms keep their CCD names unless that clashes,
    and ligand hydrogens are numbered H01, H02, ...
    """
    n_heavy = len(names)
    protein_part = set(range(len(names) - len(CAP_ATOMS) - 6, len(names)))  # N CA C O CB SG + caps
    out: Dict[int, str] = {}
    reserved = {"N", "H", "CA", "HA", "C", "O", "CB", "HB2", "HB3", "SG"}
    for i, n in enumerate(names):
        if n in CAP_ATOMS:
            continue
        if i in protein_part:
            out[i] = n
        else:
            name = n if n not in reserved else f"L{n}"[:4]
            out[i] = name
    used = set(out.values())
    h_count = 0
    hb = iter(("HB2", "HB3"))
    for atom in mol.GetAtoms():
        if atom.GetIdx() < n_heavy:
            continue
        parent = atom.GetNeighbors()[0].GetIdx()
        if parent not in out:  # a cap hydrogen
            continue
        pname = out[parent] if parent in protein_part else None
        if pname == "N":
            out[atom.GetIdx()] = "H"
        elif pname == "CA":
            out[atom.GetIdx()] = "HA"
        elif pname == "CB":
            out[atom.GetIdx()] = next(hb)
        else:
            while True:
                h_count += 1
                cand = f"H{h_count:02d}" if h_count < 100 else f"H{h_count:03d}"
                if cand not in used:
                    break
            out[atom.GetIdx()] = cand
            used.add(cand)
    if len(set(out.values())) != len(out):
        raise CovalentLigandError("atom names in the adduct residue aren't unique")
    return out


def _read_mol2_atoms(mol2: Path) -> List[Tuple[str, float]]:
    """(GAFF2 type, charge) per atom of an antechamber mol2, in file order."""
    rows, section = [], None
    for line in mol2.read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            continue
        if section == "@<TRIPOS>ATOM" and line.strip():
            f = line.split()
            rows.append((f[5], float(f[8])))
    return rows


def _residue_charges(fragment: List[Tuple[str, float]], residue: Dict[int, str],
                     net_charge: int) -> Dict[int, float]:
    """Backbone atoms take ff14SB charges, the rest their BCC charges shifted
    evenly so the residue totals `net_charge`; rounding goes onto SG."""
    charges: Dict[int, float] = {}
    rest = [i for i, n in residue.items() if n not in BACKBONE]
    backbone_q = sum(q for _, q in BACKBONE.values())
    shift = (net_charge - backbone_q - sum(fragment[i][1] for i in rest)) / len(rest)
    for i, n in residue.items():
        charges[i] = BACKBONE[n][1] if n in BACKBONE else round(fragment[i][1] + shift, 6)
    sg = next(i for i, n in residue.items() if n == "SG")
    charges[sg] = round(charges[sg] + net_charge - sum(charges.values()), 6)
    return charges


def _write_residue_mol2(path: Path, mol, residue: Dict[int, str], types: Dict[int, str],
                        charges: Dict[int, float]) -> None:
    conf = mol.GetConformer()
    order = sorted(residue)  # fragment order: ligand, then protein side, then H
    serial = {i: k + 1 for k, i in enumerate(order)}
    bonds = [b for b in mol.GetBonds()
             if b.GetBeginAtomIdx() in serial and b.GetEndAtomIdx() in serial]
    lines = ["@<TRIPOS>MOLECULE", RESNAME, f"{len(order)} {len(bonds)} 1 0 0", "SMALL", "USER_CHARGES", "",
             "@<TRIPOS>ATOM"]
    for i in order:
        p = conf.GetAtomPosition(i)
        lines.append(f"{serial[i]:7d} {residue[i]:<4s} {p.x:10.4f} {p.y:10.4f} {p.z:10.4f} "
                     f"{types[i]:<4s} 1 {RESNAME} {charges[i]:10.6f}")
    lines.append("@<TRIPOS>BOND")
    kinds = {1.0: "1", 1.5: "ar", 2.0: "2", 3.0: "3"}
    for k, b in enumerate(bonds, start=1):
        lines.append(f"{k:6d} {serial[b.GetBeginAtomIdx()]:5d} {serial[b.GetEndAtomIdx()]:5d} "
                     f"{kinds.get(b.GetBondTypeAsDouble(), '1')}")
    lines.append("@<TRIPOS>SUBSTRUCTURE")
    lines.append(f"     1 {RESNAME}  1 RESIDUE 4 A {RESNAME} 1 ****")
    path.write_text("\n".join(lines) + "\n")


def _run(cmd: Sequence[str], cwd: Path, what: str, timeout: int = 3600) -> str:
    # sqm (AM1-BCC) gains nothing from threads; see ACPYPE_OMP_THREADS
    env = {**os.environ, "OMP_NUM_THREADS": str(ACPYPE_OMP_THREADS)}
    proc = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, timeout=timeout, env=env)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        raise AmberBuildError(f"{what} failed: {out[-600:]}")
    return out


def parameterize_adduct(workspace: Path, ccd_cif: str, ligand_xyz: Dict[str, Tuple[float, float, float]],
                        bond_atom: str, residue_xyz: Dict, prev_xyz: Dict, next_xyz: Dict) -> Adduct:
    """CVX residue library and frcmod for the cysteine + bonded ligand."""
    from rdkit import Chem

    side = capped_cysteine(residue_xyz, prev_xyz, next_xyz)
    mol, rule, names = bonded_ligand(ccd_cif, ligand_xyz, bond_atom, side)
    net_charge = Chem.GetFormalCharge(mol)
    d = workspace / "adduct"
    d.mkdir(exist_ok=True)
    Chem.MolToMolFile(mol, str(d / "fragment.mol"))

    logger.info(f"covalent: AM1-BCC charges for the capped adduct ({mol.GetNumAtoms()} atoms, "
                f"rule {rule}, net charge {net_charge})")
    _run(["antechamber", "-i", "fragment.mol", "-fi", "mdl", "-o", "fragment.mol2", "-fo", "mol2",
          "-c", "bcc", "-at", "gaff2", "-nc", str(net_charge), "-rn", RESNAME, "-pf", "y"],
         d, "antechamber (AM1-BCC)")
    _run(["parmchk2", "-i", "fragment.mol2", "-f", "mol2", "-o", "fragment.frcmod", "-s", "2"],
         d, "parmchk2")
    fragment = _read_mol2_atoms(d / "fragment.mol2")
    if len(fragment) != mol.GetNumAtoms():
        raise AmberBuildError("antechamber changed the fragment's atom count")

    residue = _residue_names(mol, names)
    types = {i: BACKBONE[n][0] if n in BACKBONE else fragment[i][0] for i, n in residue.items()}
    charges = _residue_charges(fragment, residue, net_charge)
    _write_residue_mol2(d / "cvx.mol2", mol, residue, types, charges)

    script = "\n".join([
        "source leaprc.protein.ff14SB", "source leaprc.gaff2",
        "loadamberparams fragment.frcmod",
        f"{RESNAME} = loadmol2 cvx.mol2",
        f"set {RESNAME} head {RESNAME}.1.N", f"set {RESNAME} tail {RESNAME}.1.C",
        f"set {RESNAME}.1 connect0 {RESNAME}.1.N", f"set {RESNAME}.1 connect1 {RESNAME}.1.C",
        f"set {RESNAME}.1 restype protein",
        f"saveoff {RESNAME} cvx.lib", "quit", ""])
    (d / "cvx_lib.in").write_text(script)
    out = _run(["tleap", "-f", "cvx_lib.in"], d, "tleap (adduct library)")
    if not (d / "cvx.lib").exists():
        raise AmberBuildError(f"tleap wrote no adduct library: {out[-400:]}")

    conf = mol.GetConformer()
    pdb_atoms = [(residue[i], mol.GetAtomWithIdx(i).GetSymbol(), tuple(conf.GetAtomPosition(i)))
                 for i in sorted(residue)]
    n_lig = len(names) - len(CAP_ATOMS) - 6
    ligand_atoms = [residue[i] for i in sorted(residue)
                    if i < n_lig or (i >= len(names) and mol.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx() < n_lig)]
    return Adduct(d / "cvx.lib", [d / "fragment.frcmod"], pdb_atoms, ligand_atoms,
                  residue[names.index(bond_atom)], rule, net_charge)


# ---------------------------------------------------------------------------
# Junction parameters
# ---------------------------------------------------------------------------
_MISSING = {
    "bond": re.compile(r"Could not find bond parameter for atom types:\s*(\S+)\s*-\s*(\S+)"),
    "angle": re.compile(r"Could not find angle parameter for atom types:\s*(\S+)\s*-\s*(\S+)\s*-\s*(\S+)"),
    "dihedral": re.compile(r"No torsion terms for atom types:\s*([^-\s]+)-([^-\s]+)-([^-\s]+)-([^-\s]+)"),
}


def missing_parameters(leap_log: str) -> Dict[str, List[Tuple[str, ...]]]:
    out: Dict[str, List[Tuple[str, ...]]] = {k: [] for k in _MISSING}
    for kind, rx in _MISSING.items():
        for m in rx.finditer(leap_log):
            key = tuple(m.groups())
            if key not in out[kind] and tuple(reversed(key)) not in out[kind]:
                out[kind].append(key)
    return out


def _gaff2(t: str) -> str:
    if t in GAFF2_EQUIVALENT:
        return GAFF2_EQUIVALENT[t]
    if t.islower() or t[:1].islower():
        return t
    raise AmberBuildError(f"no GAFF2 equivalent for ff14SB type {t} at the adduct junction")


def junction_frcmod(path: Path, missing: Dict[str, List[Tuple[str, ...]]], param_files: Sequence[Path]) -> int:
    """frcmod with each missing term, taken from GAFF2 (plus `param_files`)
    with ff14SB types read as their GAFF2 equivalents. Returns the term count."""
    import parmed

    params = parmed.amber.AmberParameterSet(*[str(p) for p in param_files])
    # tleap rejects a NONBON section without a MASS one, even both empty
    lines = ["junction terms between ff14SB backbone and GAFF2 adduct, from GAFF2 equivalents",
             "MASS", "", "BOND"]
    n = 0
    for key in missing["bond"]:
        g = tuple(_gaff2(t) for t in key)
        bt = params.bond_types.get(g)
        if bt is None:
            raise AmberBuildError(f"no GAFF2 bond for {'-'.join(key)} ({'-'.join(g)})")
        lines.append(f"{key[0]:<2s}-{key[1]:<2s}  {bt.k:8.2f}  {bt.req:8.4f}")
        n += 1
    lines += ["", "ANGLE"]
    for key in missing["angle"]:
        g = tuple(_gaff2(t) for t in key)
        at = params.angle_types.get(g)
        if at is None:
            raise AmberBuildError(f"no GAFF2 angle for {'-'.join(key)} ({'-'.join(g)})")
        lines.append(f"{key[0]:<2s}-{key[1]:<2s}-{key[2]:<2s}  {at.k:8.3f}  {at.theteq:8.3f}")
        n += 1
    lines += ["", "DIHE"]
    for key in missing["dihedral"]:
        g = tuple(_gaff2(t) for t in key)
        terms = None
        for cand in (g, (WILDCARD, g[1], g[2], WILDCARD)):
            terms = params.dihedral_types.get(cand)
            if terms is not None:
                break
        if terms is None:
            raise AmberBuildError(f"no GAFF2 torsion for {'-'.join(key)} ({'-'.join(g)})")
        for k, term in enumerate(terms):
            sign = -1 if k < len(terms) - 1 else 1  # negative periodicity: more terms follow
            lines.append(f"{key[0]:<2s}-{key[1]:<2s}-{key[2]:<2s}-{key[3]:<2s}   1  {term.phi_k:8.3f}  "
                         f"{term.phase:8.3f}  {sign * term.per:5.1f}")
        n += 1
    lines += ["", "IMPROPER", "", "NONBON", "", ""]
    path.write_text("\n".join(lines))
    return n


def adduct_pdb_lines(adduct: Adduct, chain: str, seq: int, first_serial: int = 1) -> List[str]:
    """CVX residue as PDB ATOM lines (hydrogens included), for tleap's loadpdb."""
    from .amber_build import _pdb_atom_line
    out = []
    for k, (name, element, xyz) in enumerate(adduct.pdb_atoms):
        out.append(_pdb_atom_line(first_serial + k, name, RESNAME, chain, seq, xyz, element))
    return out


# ---------------------------------------------------------------------------
# Into the protein
# ---------------------------------------------------------------------------
def substitute_adduct(protein, chain: str, resnum: int, adduct: Adduct):
    """The prepared protein with the bonded cysteine replaced by the CVX
    residue (all its atoms, hydrogens included)."""
    from .amber_build import PreparedProtein

    try:
        seq = protein.residue_ids.index((chain, resnum)) + 1
    except ValueError:
        raise CovalentLigandError(f"bonded residue {chain}{resnum} isn't in the cleaned protein")
    out, done = [], False
    for line in protein.pdb_text.splitlines(keepends=True):
        if line.startswith("ATOM") and int(line[22:26]) == seq % 10000:
            if not done:
                if line[17:20] not in ("CYS", "CYX"):
                    raise CovalentLigandError(f"bonded residue {chain}{resnum} is {line[17:20]}, not a cysteine")
                out += adduct_pdb_lines(adduct, chain, seq)
                done = True
            continue
        out.append(line)
    disulfides = [ss for ss in protein.disulfides if seq not in ss]
    return PreparedProtein("".join(out), disulfides, protein.residue_ids, list(protein.notes))


def junction_parameters(workspace: Path, protein_pdb_text: str, disulfides, adduct: Adduct) -> int:
    """Have tleap list the terms between the ff14SB backbone and the GAFF2
    adduct it has no parameters for, and write them (from GAFF2) to
    adduct/junction.frcmod, appended to `adduct.frcmods`. Returns the count."""
    from .amber_build import _tleap_script

    d = adduct.lib.parent
    (d / "probe_protein.pdb").write_text(protein_pdb_text)
    script = _tleap_script(d / "probe_protein.pdb", disulfides, None,
                           extra_params=[adduct.lib] + adduct.frcmods)
    # no solvent needed to find the missing terms
    script = script.replace("solvateBox", "# solvateBox").replace("quit", "saveamberparm sys probe.prmtop probe.inpcrd\nquit")
    (d / "probe.in").write_text(script)
    proc = subprocess.run(["tleap", "-f", "probe.in"], cwd=d, capture_output=True, text=True, timeout=1200)
    log = (proc.stdout or "") + (proc.stderr or "")
    (d / "probe.log").write_text(log)
    missing = missing_parameters(log)
    n = sum(len(v) for v in missing.values())
    if n == 0:
        return 0
    amberhome = Path(os.environ.get("AMBERHOME", "/opt/micromamba/envs/amber"))
    gaff2 = amberhome / "dat" / "leap" / "parm" / "gaff2.dat"
    path = d / "junction.frcmod"
    junction_frcmod(path, missing, [gaff2] + adduct.frcmods)
    adduct.frcmods.append(path)
    logger.info(f"covalent: {n} junction terms from GAFF2: " +
                "; ".join(f"{k} {len(v)}" for k, v in missing.items() if v))
    return n


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------
def covalent_site(pdb_content: str, link) -> Dict:
    """Crystal coordinates the adduct is built from, read before cleaning
    strips HETATMs: the ligand's atoms, and the bonded cysteine with its
    neighbours' backbone (for the ACE/NME caps). `link` is an
    intake.covalent.CovalentLink. First altloc only."""
    resnum = int(link.resseq)
    ligand: Dict[str, Tuple[float, float, float]] = {}
    residues: Dict[int, Dict[str, Tuple[float, float, float]]] = {}
    for line in pdb_content.splitlines():
        if not line.startswith(("ATOM", "HETATM")) or line[16] not in " A":
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        name = line[12:16].strip()
        if line[17:20].strip() == link.ligand and line[21:22].strip() == link.ligand_chain \
                and line[22:26].strip() == link.ligand_resseq:
            ligand.setdefault(name, xyz)
        elif line.startswith("ATOM") and line[21:22].strip() == link.chain and line[26] == " ":
            num = int(line[22:26])
            if num in (resnum - 1, resnum, resnum + 1):
                residues.setdefault(num, {}).setdefault(name, xyz)
    if resnum not in residues:
        raise CovalentLigandError(f"bonded residue {link.chain}{resnum} not found")
    return {"ligand_code": link.ligand, "bond_atom": link.ligand_atom, "chain": link.chain,
            "resnum": resnum, "ligand": ligand, "residue": residues[resnum],
            "prev": residues.get(resnum - 1, {}), "next": residues.get(resnum + 1, {}),
            "link": link.label()}


def fetch_ccd(code: str, timeout: float = 30.0) -> str:
    """The ligand's Chemical Component Dictionary entry (mmCIF) from RCSB."""
    import httpx
    from .covalent_ligand import CCD_URL

    r = httpx.get(CCD_URL.format(code=code), timeout=timeout, follow_redirects=True)
    if r.status_code != 200:
        raise CovalentLigandError(f"CCD entry for {code} not available (HTTP {r.status_code})")
    return r.text


def build_covalent_for_pipeline(workspace: Path, compound_id: str, protein_pdb_text: str,
                                site: Dict, ccd_cif: str) -> Tuple[Path, Path, Dict]:
    """system_prep for a covalent complex: the adduct residue, the tleap
    system around it (ff14SB + GAFF2 + TIP3P), energy-checked, and what was
    done (saved as <cid>_build_info.json like the plain tleap build)."""
    from . import amber_build as ab

    adduct = parameterize_adduct(workspace, ccd_cif, site["ligand"], site["bond_atom"],
                                 site["residue"], site["prev"], site["next"])
    protein = ab.prepare_protein(protein_pdb_text)
    protein = substitute_adduct(protein, site["chain"], site["resnum"], adduct)
    n_junction = junction_parameters(workspace, protein.pdb_text, protein.disulfides, adduct)
    extra = [adduct.lib] + adduct.frcmods
    prmtop, inpcrd, stats = ab.build_system(workspace, compound_id, protein, None, extra_params=extra)
    gro, top, groups = ab.convert_to_gromacs(workspace, compound_id, prmtop, inpcrd, protein,
                                             adduct=(RESNAME, adduct.ligand_atoms))
    parity = ab.check_energy_parity(workspace, compound_id, prmtop, inpcrd, gro, top)

    # atom numbers (1-based) of the bond, to report its length over the trajectory
    bond = _bond_atom_numbers(gro, adduct.bond_atom)
    info = {
        "builder": "tleap",
        "force_field": ab.FORCE_FIELD,
        "ions": {"na": stats["na"], "cl": stats["cl"]},
        "disulfides": stats["disulfides"],
        "energy_parity": {t: v["diff"] for t, v in parity["terms"].items()},
        "notes": protein.notes,
        "covalent": {
            "link": site["link"],
            "ligand": site["ligand_code"],
            "residue": f"CYS {site['chain']}{site['resnum']} (simulated as {RESNAME})",
            "warhead_rule": adduct.rule,
            "ligand_net_charge": adduct.net_charge,
            "junction_terms_from_gaff2": n_junction,
            "bond_atoms": bond,
        },
    }
    ab.build_info_path(workspace, compound_id).write_text(json.dumps(info))
    return gro, top, info


def _bond_atom_numbers(gro: Path, ligand_atom: str) -> List[int]:
    """1-based atom numbers of SG and the bonded ligand atom in the (one)
    CVX residue of the built .gro."""
    sg = lig = None
    lines = gro.read_text().splitlines()
    for i, line in enumerate(lines[2:2 + int(lines[1])], start=1):
        if line[5:10].strip() != RESNAME:
            continue
        name = line[10:15].strip()
        if name == "SG":
            sg = i
        elif name == ligand_atom:
            lig = i
    if sg is None or lig is None:
        raise AmberBuildError("covalent bond atoms not found in the built system")
    return [sg, lig]
