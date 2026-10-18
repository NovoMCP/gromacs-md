"""AMBER (tleap) system builder: ff14SB + GAFF2 + TIP3P, converted to GROMACS.

A second topology source next to pdb2gmx (amber99sb-ildn) and ACPYPE. tleap
can build what pdb2gmx can't, such as lipids (Lipid21) and residues with a
covalently bound ligand, from residue libraries and frcmod files. The system
is solvated and ionized here, then converted with parmed, and the converted
topology is checked against the AMBER one by a single-point energy before any
dynamics run on it (`check_energy_parity`).

Outputs follow the soluble pipeline's file names, so energy minimization and
everything after it run unchanged:
  <cid>_solv_ions.gro / <cid>_amber.top   system for EM onward
  <cid>_build_groups.json                 protein / ligand atom ranges
"""

import io
import json
import os
import logging
import math
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

FORCE_FIELD = "AMBER ff14SB + GAFF2 (AM1-BCC) + TIP3P, built with tleap"

ION_CONC_M = 0.15
BOX_BUFFER_A = 10.0  # the soluble branch's editconf -d 1.0 -bt cubic
AVOGADRO = 6.02214076e23

# tleap residue and atom names -> the GROMACS names that gmx's
# residuetypes.dat and the analysis code expect (SOL water, NA/CL ions)
_GMX_RESIDUE = {"WAT": "SOL", "Na+": "NA", "Cl-": "CL", "K+": "K"}
_GMX_ATOM = {("WAT", "O"): "OW", ("WAT", "H1"): "HW1", ("WAT", "H2"): "HW2",
             ("Na+", "Na+"): "NA", ("Cl-", "Cl-"): "CL", ("K+", "K+"): "K"}
SOLVENT_RESIDUES = {"SOL", "NA", "CL", "K"}

# Single-point energy terms compared between sander and gmx, and the largest
# difference accepted, relative to the term (kcal/mol floor for small terms).
# Bonded and 1-4 terms are evaluated identically by both programs and must
# agree closely; the nonbonded terms depend on each program's PME and
# long-range dispersion implementation.
PARITY_TOLERANCE = {
    "bond": (1e-4, 0.5),
    "angle": (1e-4, 0.5),
    "dihedral": (1e-4, 0.5),
    "vdw_14": (1e-4, 0.5),
    "elec_14": (1e-4, 0.5),
    "vdw": (5e-3, 5.0),
    "elec": (2e-3, 20.0),
}


# vdW energy per atom above which the start structure has clashes: a solvated
# protein sits near +0.3 kcal/mol/atom (181L), clashing ones at hundreds
CLASH_VDW_PER_ATOM = 10.0


class AmberBuildError(RuntimeError):
    """The tleap build or its conversion to GROMACS failed."""


@dataclass
class LigandParams:
    """A GAFF2-parameterized ligand for tleap: mol2 with charges and types
    (coordinates are its starting pose) and the frcmod parmchk2 wrote."""
    mol2: Path
    frcmod: Path


@dataclass
class PreparedProtein:
    pdb_text: str                    # heavy atoms, AMBER residue names, TER at chain ends and breaks
    disulfides: List[Tuple[int, int]]  # sequential residue numbers (1-based) of bonded CYX pairs
    residue_ids: List[Tuple[str, int]]  # (chain, original residue number) per sequential residue
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Protein preparation
# ---------------------------------------------------------------------------
def _protonation_variants(topology, positions) -> List[Optional[str]]:
    """AMBER residue name per residue at pH 7 (HID/HIE/HIP, CYX, ...), from
    OpenMM's hydrogen placement, which picks histidine tautomers by H-bonding.
    None keeps the residue's own name."""
    from openmm.app import ForceField, Modeller

    modeller = Modeller(topology, positions)
    return modeller.addHydrogens(ForceField("amber14-all.xml"), pH=7.0)


def _pdb_atom_line(serial: int, name: str, resname: str, chain: str, resnum: int,
                   xyz_A: Sequence[float], element: str) -> str:
    padded = name if len(name) >= 4 else f" {name:<3}"
    x, y, z = xyz_A
    return (f"ATOM  {serial % 100000:5d} {padded:4s} {resname:>3s} {chain[:1] or 'A'}"
            f"{resnum % 10000:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
            f"          {element:>2s}\n")


def prepare_protein(pdb_text: str, gap_A: float = 2.0) -> PreparedProtein:
    """Protein PDB -> tleap input.

    Hydrogens are dropped (tleap adds them with its own names), histidines and
    other titratable residues get the AMBER name of their pH 7 state, bonded
    cysteines become CYX, and a TER is written at each chain end and at each
    chain break (peptide C-N over `gap_A`), so tleap caps the segments rather
    than bonding residues across a gap.
    """
    from openmm.app import PDBFile

    pdb = PDBFile(io.StringIO(pdb_text))
    topology, positions = pdb.topology, pdb.positions
    notes: List[str] = []
    try:
        variants = _protonation_variants(topology, positions)
    except Exception as e:  # unusual residues: keep names, tleap defaults HIS to HIE
        logger.warning(f"amber_build: protonation states not assigned ({e}); HIS built as HIE")
        variants = [None] * topology.getNumResidues()
        notes.append("histidine tautomers not assigned; all built as HIE")

    from openmm import unit
    xyz = [tuple(c * 10.0 for c in p.value_in_unit(unit.nanometer)) for p in positions]

    seq_of = {}  # openmm residue index -> sequential residue number
    lines: List[str] = []
    residue_ids: List[Tuple[str, int]] = []
    breaks: List[str] = []
    serial = 0
    for chain in topology.chains():
        residues = list(chain.residues())
        prev_c = None
        for res in residues:
            heavy = [a for a in res.atoms() if a.element is None or a.element.symbol != "H"]
            if not heavy:
                continue
            atoms = {a.name: a for a in heavy}
            if prev_c is not None and "N" in atoms:
                n = xyz[atoms["N"].index]
                if math.dist(prev_c, n) > gap_A:
                    lines.append("TER\n")
                    breaks.append(f"{chain.id}{res.id}")
            name = variants[res.index] or res.name
            if name == "HIS":
                name = "HIE"
            seq = len(residue_ids) + 1
            seq_of[res.index] = seq
            residue_ids.append((chain.id, int(re.sub(r"\D", "", res.id) or 0)))
            for a in heavy:
                serial += 1
                lines.append(_pdb_atom_line(serial, a.name, name, chain.id, seq,
                                            xyz[a.index], a.element.symbol if a.element else ""))
            prev_c = xyz[atoms["C"].index] if "C" in atoms else None
        lines.append("TER\n")
    lines.append("END\n")

    disulfides = sorted({
        tuple(sorted((seq_of[a1.residue.index], seq_of[a2.residue.index])))
        for a1, a2 in topology.bonds()
        if a1.name == "SG" and a2.name == "SG" and a1.residue.index != a2.residue.index
        and a1.residue.index in seq_of and a2.residue.index in seq_of
    })
    if breaks:
        notes.append(f"chain breaks before {', '.join(breaks[:8])}: segments built with charged termini")
    n_his = sum(1 for v in variants if v in ("HID", "HIE", "HIP"))
    if n_his:
        counts = {k: sum(1 for v in variants if v == k) for k in ("HID", "HIE", "HIP")}
        logger.info(f"amber_build: histidines at pH 7: {counts}")
    return PreparedProtein("".join(lines), disulfides, residue_ids, notes)


# ---------------------------------------------------------------------------
# tleap
# ---------------------------------------------------------------------------
def _tleap_script(protein_pdb: Optional[Path], disulfides: Sequence[Tuple[int, int]],
                  ligand: Optional[LigandParams], extra_leaprc: Sequence[str] = (),
                  ions: Optional[Tuple[int, int]] = None, out_prefix: Optional[str] = None,
                  extra_params: Sequence[Path] = ()) -> str:
    s = ["source leaprc.protein.ff14SB", "source leaprc.water.tip3p", "source leaprc.gaff2"]
    s += [f"source {rc}" for rc in extra_leaprc]
    # residue libraries and frcmods, e.g. a covalent adduct residue
    s += [f"loadoff {p.name}" if p.suffix == ".lib" else f"loadamberparams {p.name}" for p in extra_params]
    parts = []
    if ligand is not None:
        s.append(f"loadamberparams {ligand.frcmod.name}")
        s.append(f"lig = loadmol2 {ligand.mol2.name}")
    if protein_pdb is not None:
        s.append(f"prot = loadpdb {protein_pdb.name}")
        s += [f"bond prot.{i}.SG prot.{j}.SG" for i, j in disulfides]
        parts.append("prot")
    if ligand is not None:
        parts.append("lig")
    s.append(f"sys = combine {{ {' '.join(parts)} }}")
    s.append(f"solvateBox sys TIP3PBOX {BOX_BUFFER_A} iso")
    s.append("charge sys")
    if ions is not None:
        n_na, n_cl = ions
        spec = " ".join(f"{ion} {n}" for ion, n in (("Na+", n_na), ("Cl-", n_cl)) if n > 0)
        if spec:
            s.append(f"addIonsRand sys {spec}")
        s.append(f"saveamberparm sys {out_prefix}.prmtop {out_prefix}.inpcrd")
        s.append(f"savepdb sys {out_prefix}_leap.pdb")
    s.append("quit")
    return "\n".join(s) + "\n"


def _leap_errors(log: str) -> List[str]:
    errs = []
    for line in log.splitlines():
        low = line.lower()
        if "fatal" in low or low.startswith("error") or "could not open" in low \
                or "does not have a type" in low or "created a new atom named" in low \
                or "could not find" in low or "could not load" in low \
                or "parameter file was not saved" in low:
            errs.append(line.strip())
    return errs


def _run_tleap(workspace: Path, script: str, name: str) -> str:
    path = workspace / f"{name}.in"
    path.write_text(script)
    proc = subprocess.run(["tleap", "-f", path.name], cwd=workspace,
                          capture_output=True, text=True, timeout=1200)
    log = (proc.stdout or "") + (proc.stderr or "")
    (workspace / f"{name}.log").write_text(log)
    errors = _leap_errors(log)
    if proc.returncode != 0 or errors:
        raise AmberBuildError(f"tleap {name}: " + ("; ".join(errors[:5]) or log[-600:]))
    return log


def _parse_volume_and_charge(log: str) -> Tuple[float, float]:
    """Box volume (Å³) that solvateBox reports, and the system's net charge."""
    volume = re.findall(r"Volume:\s+([\d.]+)\s+A\^3", log)
    charge = re.findall(r"Total unperturbed charge:\s+(-?[\d.]+)", log)
    if not volume or not charge:
        raise AmberBuildError("tleap: box volume or total charge not found in its log")
    return float(volume[-1]), float(charge[-1])


def ion_counts(volume_A3: float, net_charge: float,
               conc_M: float = ION_CONC_M) -> Tuple[int, int]:
    """(Na+, Cl-) to neutralize and reach `conc_M` NaCl over the box volume,
    as `gmx genion -neutral -conc` does on the soluble branch."""
    volume_L = volume_A3 * 1e-27
    pairs = int(round(conc_M * AVOGADRO * volume_L))
    q = int(round(net_charge))
    return pairs + max(-q, 0), pairs + max(q, 0)


def build_system(workspace: Path, compound_id: str, protein: Optional[PreparedProtein] = None,
                 ligand: Optional[LigandParams] = None,
                 extra_params: Sequence[Path] = ()) -> Tuple[Path, Path, Dict]:
    """Solvated, ionized AMBER system: prmtop and inpcrd, and build stats.

    tleap runs twice: the first build reads the box volume and net charge,
    the second adds the ions those imply and writes the topology.
    """
    if protein is None and ligand is None:
        raise ValueError("build_system needs a protein, a ligand, or both")
    protein_pdb = None
    if protein is not None:
        protein_pdb = workspace / f"{compound_id}_protein_amber.pdb"
        protein_pdb.write_text(protein.pdb_text)
    for p in ([ligand.mol2, ligand.frcmod] if ligand is not None else []) + list(extra_params):
        if p.parent != workspace:
            shutil.copy(p, workspace / p.name)

    disulfides = protein.disulfides if protein else []
    log = _run_tleap(workspace, _tleap_script(protein_pdb, disulfides, ligand,
                                              extra_params=extra_params), "leap_probe")
    volume, charge = _parse_volume_and_charge(log)
    n_na, n_cl = ion_counts(volume, charge)

    prefix = f"{compound_id}_amber"
    _run_tleap(workspace, _tleap_script(protein_pdb, disulfides, ligand, ions=(n_na, n_cl),
                                        out_prefix=prefix, extra_params=extra_params), "leap_build")
    prmtop, inpcrd = workspace / f"{prefix}.prmtop", workspace / f"{prefix}.inpcrd"
    if not prmtop.exists() or prmtop.stat().st_size == 0:
        raise AmberBuildError("tleap wrote no prmtop")
    stats = {"volume_nm3": round(volume / 1000.0, 1), "net_charge": round(charge, 3),
             "na": n_na, "cl": n_cl, "disulfides": len(disulfides)}
    logger.info(f"amber_build: tleap system built: {stats}")
    return prmtop, inpcrd, stats


# ---------------------------------------------------------------------------
# Conversion to GROMACS
# ---------------------------------------------------------------------------
def _renumber(structure, protein: Optional[PreparedProtein]) -> None:
    """Protein residues keep their PDB numbers (as pdb2gmx leaves them), the
    ligand follows the last protein residue, and solvent follows on."""
    n_prot = len(protein.residue_ids) if protein else 0
    last = 0
    for i, res in enumerate(structure.residues):
        if i < n_prot:
            chain, num = protein.residue_ids[i]
            res.number, res.chain = num, chain
            last = num
        else:
            last += 1
            res.number = last


def _gmx_names(structure) -> None:
    for res in structure.residues:
        old = res.name
        for atom in res.atoms:
            atom.name = _GMX_ATOM.get((old, atom.name), atom.name)
        res.name = _GMX_RESIDUE.get(old, old)


_ATOM_RESNR = re.compile(r"(\s*\S+\s+\S+\s+)(\d+)")


def _renumber_atoms_sections(top_text: str, numbers: List[List[int]], names: List[List[str]]) -> str:
    """Set the resnr column of each moleculetype's [ atoms ], the k-th
    moleculetype taking numbers[k] (by the molecule's own residue index)."""
    out, k, section = [], -1, None
    for line in top_text.splitlines():
        stripped = line.split(";")[0].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip()
            if section == "moleculetype":
                k += 1
        elif section == "atoms" and stripped and not stripped.startswith("#") and 0 <= k < len(numbers):
            m = _ATOM_RESNR.match(line)  # nr, type, then resnr
            local, resname = int(m.group(2)) - 1, line.split()[3]
            if names[k][local] != resname:
                raise AmberBuildError(f"residue renumbering out of step at {resname} "
                                      f"(expected {names[k][local]})")
            line = f"{m.group(1)}{numbers[k][local]:>{len(m.group(2))}}{line[m.end(2):]}"
        out.append(line)
    return "\n".join(out) + "\n"


def apply_residue_numbers(structure, top: Path, gro: Path) -> None:
    """Write the residues' own numbers (PDB numbers for the protein, see
    _renumber) into the .top and .gro. parmed writes each residue's index
    instead, and GROMACS takes residue numbers from the topology, so without
    this every label after the build (H-bonds, RMSF, decomposition) would be
    a sequence position, not the residue the user knows."""
    molecules = structure.split()  # the order parmed writes the moleculetypes in
    numbers = [[r.number for r in mol.residues] for mol, _ in molecules]
    names = [[r.name for r in mol.residues] for mol, _ in molecules]
    top.write_text(_renumber_atoms_sections(top.read_text(), numbers, names))
    lines = gro.read_text().splitlines()
    for i, atom in enumerate(structure.atoms):
        line = lines[2 + i]
        lines[2 + i] = f"{atom.residue.number % 100000:5d}{line[5:]}"
    gro.write_text("\n".join(lines) + "\n")


def heavy_atom_posre(atoms_section: Sequence[str], k: int = 1000) -> List[str]:
    """[ position_restraints ] on the heavy atoms of one [ atoms ] section."""
    out = ["[ position_restraints ]", ";  ai  funct    fcx    fcy    fcz"]
    for line in atoms_section:
        fields = line.split(";")[0].split()
        if len(fields) < 8:
            continue
        if float(fields[7]) >= 1.5:  # not hydrogen
            out.append(f"{int(fields[0]):6d}      1   {k}   {k}   {k}")
    return out


def add_position_restraints(top_text: str, skip_residues=SOLVENT_RESIDUES, k: int = 1000) -> str:
    """Add `#ifdef POSRES` heavy-atom restraints to each solute moleculetype.

    parmed writes no posre files; pdb2gmx and the ligand step restrain solute
    heavy atoms at 1000 kJ/mol/nm² during NVT and NPT, and so does this.
    Moleculetypes made only of water and ions are left free.
    """
    lines = top_text.splitlines()
    out: List[str] = []
    section = None
    atoms: List[str] = []
    in_mol = False

    def flush():
        resnames = {l.split(";")[0].split()[3] for l in atoms if len(l.split(";")[0].split()) >= 8}
        if atoms and not resnames <= set(skip_residues):
            out.extend(["", "#ifdef POSRES"] + heavy_atom_posre(atoms, k) + ["#endif", ""])

    for line in lines:
        stripped = line.split(";")[0].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            new = stripped[1:-1].strip()
            if new in ("moleculetype", "system") and in_mol:
                flush()
                atoms = []
                in_mol = False
            if new == "moleculetype":
                in_mol = True
            section = new
        elif in_mol and section == "atoms" and stripped:
            atoms.append(line)
        out.append(line)
    if in_mol:
        flush()
    return "\n".join(out) + "\n"


def _ranges(indices: Sequence[int]) -> List[List[int]]:
    out: List[List[int]] = []
    for i in indices:
        if out and out[-1][1] == i - 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def write_build_groups(structure, path: Path, ligand_resnames: Sequence[str] = (),
                       n_protein_residues: int = 0,
                       adduct: Optional[Tuple[str, Sequence[str]]] = None) -> Dict:
    """Protein and ligand atoms (1-based ranges), from what the build put
    where, so analysis doesn't classify atoms by residue name. `adduct` is
    (residue name, ligand atom names) for a ligand bonded into a protein
    residue: those atoms go to the ligand, the residue's others to the protein."""
    protein, ligand = [], []
    for i, res in enumerate(structure.residues):
        if adduct is not None and res.name == adduct[0]:
            for a in res.atoms:
                (ligand if a.name in adduct[1] else protein).append(a.idx + 1)
            continue
        idx = [a.idx + 1 for a in res.atoms]
        if i < n_protein_residues:
            protein += idx
        elif res.name in ligand_resnames:
            ligand += idx
    groups = {"protein": _ranges(protein), "ligand": _ranges(ligand),
              "ligand_resnames": sorted(set(ligand_resnames))}
    path.write_text(json.dumps(groups))
    return groups


def convert_to_gromacs(workspace: Path, compound_id: str, prmtop: Path, inpcrd: Path,
                       protein: Optional[PreparedProtein] = None,
                       adduct: Optional[Tuple[str, Sequence[str]]] = None) -> Tuple[Path, Path, Dict]:
    """prmtop/inpcrd -> <cid>_solv_ions.gro and <cid>_amber.top (with
    POSRES blocks) and <cid>_build_groups.json."""
    import parmed

    structure = parmed.load_file(str(prmtop), xyz=str(inpcrd))
    n_prot = len(protein.residue_ids) if protein else 0
    ligand_resnames = sorted({r.name for r in structure.residues[n_prot:]
                              if r.name not in _GMX_RESIDUE})
    _renumber(structure, protein)
    _gmx_names(structure)

    gro = workspace / f"{compound_id}_solv_ions.gro"
    top = workspace / f"{compound_id}_amber.top"
    gmx_top = parmed.gromacs.GromacsTopologyFile.from_structure(structure)
    gmx_top.write(str(top))
    top.write_text(add_position_restraints(top.read_text()))
    structure.save(str(gro), format="gro", overwrite=True)
    apply_residue_numbers(structure, top, gro)
    groups = write_build_groups(structure, workspace / f"{compound_id}_build_groups.json",
                                ligand_resnames, n_prot, adduct)
    return gro, top, groups


# ---------------------------------------------------------------------------
# Energy parity: sander on the AMBER files vs gmx on the converted ones
# ---------------------------------------------------------------------------
_KJ_PER_KCAL = 4.184

_SANDER_IN = """single point
 &cntrl
  imin=0, nstlim=0, ntx=1, irest=0, ntpr=1, ntb=1, cut={cut},
  ntc=1, ntf=1, ig=1,
 /
"""

# FLEXIBLE: parmed writes water as SETTLE unless it's defined, and sander
# (ntf=1) evaluates the water bonds
_GMX_PARITY_MDP = """\
define          = -DFLEXIBLE
integrator      = md
nsteps          = 0
nstcalcenergy   = 1
nstenergy       = 1
cutoff-scheme   = Verlet
pbc             = xyz
coulombtype     = PME
rcoulomb        = {cut_nm}
fourierspacing  = 0.1
pme-order       = 6
ewald-rtol      = 1e-5
vdwtype         = Cut-off
vdw-modifier    = None
rvdw            = {cut_nm}
coulomb-modifier = None
DispCorr        = Ener
constraints     = none
verlet-buffer-tolerance = -1
rlist           = {cut_nm}
"""

# sander mdout labels and gmx energy term names for each compared term
_SANDER_TERMS = {"bond": ["BOND"], "angle": ["ANGLE"], "dihedral": ["DIHED"],
                 "vdw_14": ["1-4 NB"], "elec_14": ["1-4 EEL"], "vdw": ["VDWAALS"],
                 "elec": ["EELEC"]}
_GMX_TERMS = {"bond": ["Bond"], "angle": ["Angle", "U-B"],
              "dihedral": ["Proper Dih.", "Improper Dih.", "Ryckaert-Bell.", "Per. Imp. Dih."],
              "vdw_14": ["LJ-14"], "elec_14": ["Coulomb-14"],
              "vdw": ["LJ (SR)", "Disper. corr."],
              "elec": ["Coulomb (SR)", "Coul. recip."]}


def _parse_sander(mdout: str) -> Dict[str, float]:
    block = mdout.split("NSTEP =")[1] if "NSTEP =" in mdout else mdout
    values = {}
    for m in re.finditer(r"([A-Z0-9][A-Z0-9 \-]*?)\s*=\s*(-?\d+\.\d+)", block):
        values[m.group(1).strip()] = float(m.group(2))
    out = {}
    for term, labels in _SANDER_TERMS.items():
        if all(l in values for l in labels):
            out[term] = sum(values[l] for l in labels)
    return out


def _parse_gmx_log(log: str) -> Dict[str, float]:
    """Energies of the last 'Energies' block of an mdrun log: header rows of
    15-character fields, each followed by a row of values."""
    blocks = log.split("   Energies (kJ/mol)")
    if len(blocks) < 2:
        raise AmberBuildError("gmx parity run: no energies in md.log")
    rows = blocks[-1].split("\n\n")[0].splitlines()[1:]
    values = {}
    for header, vals in zip(rows[0::2], rows[1::2]):
        names = [header[i:i + 15].strip() for i in range(0, len(header), 15)]
        nums = [vals[i:i + 15].strip() for i in range(0, len(vals), 15)]
        for n, v in zip(names, nums):
            if n and v:
                values[n] = float(v)
    out = {}
    for term, names in _GMX_TERMS.items():
        present = [values[n] for n in names if n in values]
        if present:
            out[term] = sum(present) / _KJ_PER_KCAL
    return out


def _gmx(cmd: Sequence[str], cwd: Path) -> None:
    # mdrun refuses an OMP_NUM_THREADS that differs from -ntomp
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    p = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, timeout=1200, env=env)
    if p.returncode != 0:
        raise AmberBuildError(f"gmx parity run failed ({cmd[1]}): {p.stderr[-400:]}")


def _single_points(pdir: Path, prmtop: Path, inpcrd: Path, gro: Path, top: Path,
                   cut_A: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """(sander, gmx) energy terms at the coordinates of `gro`."""
    import parmed

    # Both programs evaluate the .gro coordinates: rounded to 0.01 Å there,
    # they shift the bond and angle energies by more than the tolerance
    structure = parmed.load_file(str(prmtop), xyz=str(inpcrd))
    from_gro = parmed.load_file(str(gro))
    structure.coordinates = from_gro.coordinates
    structure.box = from_gro.box
    structure.save(str(pdir / "sp.rst7"), format="rst7", overwrite=True)
    (pdir / "sp.in").write_text(_SANDER_IN.format(cut=cut_A))
    proc = subprocess.run(
        ["sander", "-O", "-i", "sp.in", "-o", "sp.out", "-p", str(prmtop.resolve()),
         "-c", "sp.rst7", "-r", "sp.rst", "-inf", "sp.info"],
        cwd=pdir, capture_output=True, text=True, timeout=1200,
    )
    mdout = (pdir / "sp.out").read_text() if (pdir / "sp.out").exists() else ""
    if proc.returncode != 0 or "NSTEP" not in mdout:
        raise AmberBuildError(f"sander single point failed: {(proc.stderr or mdout)[-400:]}")

    (pdir / "sp.mdp").write_text(_GMX_PARITY_MDP.format(cut_nm=cut_A / 10.0))
    _gmx(["gmx", "grompp", "-f", "sp.mdp", "-c", str(gro.resolve()), "-p", str(top.resolve()),
          "-o", "sp.tpr", "-maxwarn", "5"], pdir)
    _gmx(["gmx", "mdrun", "-s", "sp.tpr", "-deffnm", "sp", "-nb", "cpu", "-pme", "cpu",
          "-ntmpi", "1", "-ntomp", "4"], pdir)
    return _parse_sander(mdout), _parse_gmx_log((pdir / "sp.log").read_text())


_RELAX_MDP = """\
integrator = steep
nsteps     = 2000
emtol      = 1000
cutoff-scheme = Verlet
coulombtype = PME
rcoulomb   = 1.0
rvdw       = 1.0
"""


def _relax(pdir: Path, gro: Path, top: Path) -> Path:
    (pdir / "relax.mdp").write_text(_RELAX_MDP)
    _gmx(["gmx", "grompp", "-f", "relax.mdp", "-c", str(gro.resolve()), "-p", str(top.resolve()),
          "-o", "relax.tpr", "-maxwarn", "5"], pdir)
    _gmx(["gmx", "mdrun", "-s", "relax.tpr", "-deffnm", "relax", "-nb", "cpu", "-pme", "cpu",
          "-ntmpi", "1", "-ntomp", "4"], pdir)
    return pdir / "relax.gro"


def check_energy_parity(workspace: Path, compound_id: str, prmtop: Path, inpcrd: Path,
                        gro: Path, top: Path, cut_A: float = 10.0,
                        tolerance: Dict[str, Tuple[float, float]] = PARITY_TOLERANCE) -> Dict:
    """Single-point energy of the AMBER system (sander) and of the converted
    GROMACS system (gmx), term by term, in kcal/mol. Raises AmberBuildError if
    a term differs by more than its tolerance: a conversion error is otherwise
    silent, and the simulation would run on a different force field."""
    pdir = workspace / "parity"
    pdir.mkdir(exist_ok=True)
    amber, gmx = _single_points(pdir, prmtop, inpcrd, gro, top, cut_A)
    relaxed = False
    n_atoms = int(gro.read_text().splitlines()[1])
    if set(_SANDER_TERMS) - set(amber) or amber.get("vdw", 0.0) > CLASH_VDW_PER_ATOM * n_atoms:
        # Atoms rebuilt by PDBFixer can start overlapping: the vdW energy is
        # then orders of magnitude off (or too large for sander to print), and
        # a relative tolerance on it means nothing. Minimize briefly with the
        # converted topology and compare both programs at those coordinates.
        logger.info("amber_build: start structure clashes; comparing energies after a short minimization")
        gro = _relax(pdir, gro, top)
        amber, gmx = _single_points(pdir, prmtop, inpcrd, gro, top, cut_A)
        relaxed = True

    terms, failed = {}, []
    for term, (rel, floor) in tolerance.items():
        if term not in amber or term not in gmx:
            failed.append(f"{term}: missing ({'sander' if term not in amber else 'gmx'})")
            continue
        a, g = amber[term], gmx[term]
        diff = g - a
        ok = abs(diff) <= max(rel * abs(a), floor)
        terms[term] = {"amber": round(a, 3), "gromacs": round(g, 3), "diff": round(diff, 3), "ok": ok}
        if not ok:
            failed.append(f"{term}: amber {a:.2f} vs gromacs {g:.2f} kcal/mol")
    result = {"terms": terms, "passed": not failed, "after_minimization": relaxed}
    (workspace / f"{compound_id}_energy_parity.json").write_text(json.dumps(result, indent=1))
    if failed:
        raise AmberBuildError("AMBER -> GROMACS conversion changed the energy: " + "; ".join(failed))
    logger.info("amber_build: energy parity passed: " +
                ", ".join(f"{t} {v['diff']:+.3f}" for t, v in terms.items()))
    return result


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------
BUILDERS = ("pdb2gmx", "tleap")
PDB2GMX_FORCE_FIELD = "AMBER99SB-ILDN + GAFF2 (AM1-BCC) + TIP3P, built with pdb2gmx"


def build_info_path(workspace: Path, compound_id: str) -> Path:
    return workspace / f"{compound_id}_build_info.json"


def build_for_pipeline(workspace: Path, compound_id: str, protein_pdb_text: Optional[str],
                       ligand_pdb: Optional[Path], ligand_smiles: Optional[str]) -> Tuple[Path, Path, Dict]:
    """system_prep for the tleap builder: the solvated, ionized system as
    <cid>_solv_ions.gro + <cid>_amber.top, energy-checked, and what the build
    did (also saved to <cid>_build_info.json for a resumed job's result)."""
    from .topology import generate_ligand_topology

    protein = prepare_protein(protein_pdb_text) if protein_pdb_text else None
    ligand = None
    if ligand_pdb is not None:
        # ACPYPE's AM1-BCC charges and GAFF2 types, as on the pdb2gmx branch
        generate_ligand_topology(workspace, compound_id, ligand_pdb, ligand_smiles)
        base = compound_id.replace("-", "_")
        acpype_dir = workspace / f"{base}.acpype"
        ligand = LigandParams(acpype_dir / f"{base}_bcc_gaff2.mol2", acpype_dir / f"{base}_AC.frcmod")
    prmtop, inpcrd, stats = build_system(workspace, compound_id, protein, ligand)
    gro, top, groups = convert_to_gromacs(workspace, compound_id, prmtop, inpcrd, protein)
    parity = check_energy_parity(workspace, compound_id, prmtop, inpcrd, gro, top)
    info = {
        "builder": "tleap",
        "force_field": FORCE_FIELD,
        "ions": {"na": stats["na"], "cl": stats["cl"]},
        "disulfides": stats["disulfides"],
        "energy_parity": {t: v["diff"] for t, v in parity["terms"].items()},
        "notes": protein.notes if protein else [],
    }
    build_info_path(workspace, compound_id).write_text(json.dumps(info))
    return gro, top, info


def load_build_info(workspace: Path, compound_id: str) -> Dict:
    path = build_info_path(workspace, compound_id)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except ValueError:
            pass
    return {"builder": "pdb2gmx", "force_field": PDB2GMX_FORCE_FIELD}
