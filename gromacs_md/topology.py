"""GROMACS topology generation and MDP file writers."""

import logging
import subprocess
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def process_protein_with_gromacs(workspace: Path, compound_id: str, pdb_path: Path):
    """Process protein with GROMACS pdb2gmx (AMBER99SB-ILDN force field)."""
    gro_file = workspace / f"{compound_id}_protein.gro"
    top_file = workspace / f"{compound_id}_protein.top"

    result = subprocess.run(
        [
            "gmx", "pdb2gmx",
            "-f", str(pdb_path),
            "-o", str(gro_file),
            "-p", str(top_file),
            "-water", "tip3p",
            "-ff", "amber99sb-ildn",
            "-ignh",
        ],
        input=b"1\n",
        capture_output=True,
        cwd=workspace,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace") if result.stderr else ""
        logger.error(f"pdb2gmx failed:\n{stderr}")
        raise subprocess.CalledProcessError(result.returncode, "pdb2gmx", stderr=result.stderr)
    return gro_file, top_file


def generate_ligand_topology(workspace: Path, compound_id: str, ligand_pdb: Path, ligand_smiles: str):
    """Generate GROMACS topology for ligand using OpenBabel + ACPYPE."""
    # ACPYPE doesn't handle hyphens well in base names — normalize
    acpype_base = compound_id.replace("-", "_")
    ligand_mol2 = workspace / f"{acpype_base}_ligand.mol2"

    result = subprocess.run(
        ["obabel", str(ligand_pdb), "-O", str(ligand_mol2), "-h"],
        capture_output=True, cwd=workspace,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace") if result.stderr else ""
        logger.error(f"obabel failed:\n{stderr}")
        raise subprocess.CalledProcessError(result.returncode, "obabel", stderr=result.stderr)

    # ACPYPE expects a 'tmp' directory in cwd
    (workspace / "tmp").mkdir(exist_ok=True)

    logger.info("Running ACPYPE for ligand parameterization...")
    result = subprocess.run(
        ["acpype", "-i", str(ligand_mol2), "-b", acpype_base, "-c", "bcc", "-a", "gaff2"],
        capture_output=True, cwd=workspace,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace") if result.stderr else ""
        logger.error(f"ACPYPE failed:\n{stderr}")
        raise subprocess.CalledProcessError(result.returncode, "acpype", stderr=result.stderr)

    acpype_dir = workspace / f"{acpype_base}.acpype"
    ligand_gro = acpype_dir / f"{acpype_base}_GMX.gro"
    ligand_itp = acpype_dir / f"{acpype_base}_GMX.itp"

    shutil.copy(ligand_gro, workspace / f"{compound_id}_ligand.gro")
    shutil.copy(ligand_itp, workspace / f"{compound_id}_ligand.itp")

    return workspace / f"{compound_id}_ligand.gro", workspace / f"{compound_id}_ligand.itp"


def combine_protein_ligand(
    workspace: Path,
    compound_id: str,
    protein_gro: Path,
    ligand_gro: Path,
    protein_top: Path,
    ligand_itp: Path,
):
    """Combine protein and ligand GRO/topology into complex."""
    complex_gro = workspace / f"{compound_id}_complex.gro"
    complex_top = workspace / f"{compound_id}_complex.top"

    with open(protein_gro, "r") as f:
        protein_lines = f.readlines()
    with open(ligand_gro, "r") as f:
        ligand_lines = f.readlines()

    protein_atoms = int(protein_lines[1].strip())
    ligand_atoms = int(ligand_lines[1].strip())
    total_atoms = protein_atoms + ligand_atoms

    with open(complex_gro, "w") as f:
        f.write(protein_lines[0])
        f.write(f"{total_atoms:5d}\n")
        for line in protein_lines[2 : 2 + protein_atoms]:
            f.write(line)
        last_resnum = int(protein_lines[2 + protein_atoms - 1][0:5])
        for line in ligand_lines[2 : 2 + ligand_atoms]:
            new_line = f"{last_resnum + 1:5d}" + line[5:]
            f.write(new_line)
        f.write(protein_lines[-1])

    # Extract [ atomtypes ] from ligand itp (must go before any [ moleculetype ])
    with open(ligand_itp, "r") as f:
        itp_content = f.read()

    atomtypes_section = ""
    cleaned_itp_lines = []
    in_atomtypes = False
    for line in itp_content.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "[ atomtypes ]":
            in_atomtypes = True
            atomtypes_section += line
        elif in_atomtypes and stripped.startswith("[") and stripped != "[ atomtypes ]":
            in_atomtypes = False
            cleaned_itp_lines.append(line)
        elif in_atomtypes:
            atomtypes_section += line
        else:
            cleaned_itp_lines.append(line)

    # Write cleaned itp
    with open(ligand_itp, "w") as f:
        f.writelines(cleaned_itp_lines)

    # Read the molecule name from the ligand ITP [ moleculetype ] section
    ligand_mol_name = compound_id
    with open(ligand_itp, "r") as f:
        in_moltype = False
        for line in f:
            if "[ moleculetype ]" in line:
                in_moltype = True
                continue
            if in_moltype and line.strip() and not line.startswith(";"):
                ligand_mol_name = line.split()[0]
                break

    with open(complex_top, "w") as f:
        with open(protein_top, "r") as pf:
            atomtypes_written = False
            for line in pf:
                f.write(line)
                # Insert ligand atomtypes right after forcefield.itp include
                if not atomtypes_written and atomtypes_section and "forcefield.itp" in line:
                    f.write("\n; Ligand atom types from ACPYPE/GAFF2\n")
                    f.write(atomtypes_section + "\n")
                    atomtypes_written = True
                # Insert ligand itp include and molecule entry at [ molecules ]
                if "[ molecules ]" in line:
                    # Write ligand itp include before molecule list
                    # (already wrote [ molecules ] line above)
                    # Read remaining lines to find protein mol entry and append ligand
                    for mol_line in pf:
                        f.write(mol_line)
                        if mol_line.strip() and not mol_line.startswith(";"):
                            # This is the protein molecule entry — add ligand after it
                            f.write(f"{ligand_mol_name}               1\n")
                            break
                    # Also need the ligand itp include — place before [ system ]
        # Now re-read and insert ligand itp include in right place
        # Actually, we need to add the include above [ system ] section

    # Re-read complex_top and insert ligand #include before [ system ]
    with open(complex_top, "r") as f:
        top_lines = f.readlines()

    with open(complex_top, "w") as f:
        for line in top_lines:
            if "[ system ]" in line:
                f.write(f'; Include ligand topology\n')
                f.write(f'#include "{ligand_itp.name}"\n\n')
            f.write(line)

    return complex_gro, complex_top


def create_ligand_only_topology(workspace: Path, compound_id: str, ligand_itp: Path):
    """Create standalone topology for ligand-only simulation.

    GROMACS requires [ atomtypes ] before any [ moleculetype ]. ACPYPE puts
    atomtypes in the .itp file, so we extract them into the .top file and
    write a cleaned .itp without atomtypes.
    """
    # Read ligand itp and split out [ atomtypes ]
    with open(ligand_itp, "r") as f:
        itp_content = f.read()

    atomtypes_section = ""
    cleaned_itp_lines = []
    in_atomtypes = False
    for line in itp_content.splitlines(keepends=True):
        stripped = line.strip()
        if stripped == "[ atomtypes ]":
            in_atomtypes = True
            atomtypes_section += line
        elif in_atomtypes and stripped.startswith("[") and stripped != "[ atomtypes ]":
            in_atomtypes = False
            cleaned_itp_lines.append(line)
        elif in_atomtypes:
            atomtypes_section += line
        else:
            cleaned_itp_lines.append(line)

    # Write cleaned itp (without atomtypes)
    with open(ligand_itp, "w") as f:
        f.writelines(cleaned_itp_lines)

    # Read the actual molecule name from the ITP [ moleculetype ] section
    # (ACPYPE uses names like "MOL" or the sanitized molecule name, not compound_id)
    ligand_mol_name = compound_id
    in_moltype = False
    for line in cleaned_itp_lines:
        if "[ moleculetype ]" in line:
            in_moltype = True
            continue
        if in_moltype and line.strip() and not line.startswith(";"):
            ligand_mol_name = line.split()[0]
            break

    top_file = workspace / f"{compound_id}.top"
    with open(top_file, "w") as f:
        f.write(f"; Topology for {compound_id} ligand-only simulation\n")
        f.write('#include "amber99sb-ildn.ff/forcefield.itp"\n\n')
        # Atomtypes must come before any moleculetype
        if atomtypes_section:
            f.write("; Ligand atom types from ACPYPE/GAFF2\n")
            f.write(atomtypes_section + "\n")
        f.write('#include "amber99sb-ildn.ff/tip3p.itp"\n\n')
        f.write(f'#include "{ligand_itp.name}"\n\n')
        f.write("[ system ]\n")
        f.write(f"{compound_id}\n\n")
        f.write("[ molecules ]\n")
        f.write(f"{ligand_mol_name}    1\n")
    return top_file


# ---------------------------------------------------------------------------
# MDP File Generators
# ---------------------------------------------------------------------------
def write_em_mdp(workspace: Path, compound_id: str):
    content = """\
integrator  = steep
emtol       = 500.0
emstep      = 0.01
nsteps      = 100000
nstlist     = 1
cutoff-scheme = Verlet
ns_type     = grid
coulombtype = PME
rcoulomb    = 1.0
rvdw        = 1.0
pbc         = xyz
"""
    with open(workspace / f"{compound_id}_em.mdp", "w") as f:
        f.write(content)


def write_nvt_mdp(workspace: Path, compound_id: str, temperature: float):
    content = f"""\
integrator              = md
nsteps                  = 100000
dt                      = 0.001
nstxout                 = 1000
nstvout                 = 1000
nstenergy               = 1000
nstlog                  = 1000
tcoupl                  = V-rescale
tc-grps                 = System
tau_t                   = 0.1
ref_t                   = {temperature}
pcoupl                  = no
nstlist                 = 10
cutoff-scheme           = Verlet
ns_type                 = grid
coulombtype             = PME
rcoulomb                = 1.0
rvdw                    = 1.0
constraints             = h-bonds
constraint_algorithm    = lincs
lincs-iter              = 2
lincs-order             = 6
"""
    with open(workspace / f"{compound_id}_nvt.mdp", "w") as f:
        f.write(content)


def write_npt_mdp(workspace: Path, compound_id: str, temperature: float, pressure: float, nsteps: int = 100000):
    # nsteps default 100000 (= 100 ps at dt=0.001) preserves the legacy
    # pipeline behavior. Callers running adaptive equilibration override
    # this with a smaller initial window (typically 50000 = 50 ps) and
    # then extend via gmx convert-tpr -extend on subsequent passes.
    content = f"""\
integrator              = md
nsteps                  = {nsteps}
dt                      = 0.001
nstxout                 = 1000
nstvout                 = 1000
nstenergy               = 1000
nstlog                  = 1000
tcoupl                  = V-rescale
tc-grps                 = System
tau_t                   = 0.1
ref_t                   = {temperature}
pcoupl                  = Parrinello-Rahman
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = {pressure}
compressibility         = 4.5e-5
nstlist                 = 10
cutoff-scheme           = Verlet
ns_type                 = grid
coulombtype             = PME
rcoulomb                = 1.0
rvdw                    = 1.0
constraints             = h-bonds
constraint_algorithm    = lincs
lincs-iter              = 2
lincs-order             = 6
"""
    with open(workspace / f"{compound_id}_npt.mdp", "w") as f:
        f.write(content)


def write_md_mdp(workspace: Path, compound_id: str, simulation_ns: float, temperature: float, pressure: float):
    nsteps = int(simulation_ns * 500000)
    content = f"""\
integrator              = md
nsteps                  = {nsteps}
dt                      = 0.002
nstxout                 = 0
nstvout                 = 0
nstfout                 = 0
nstlog                  = 5000
nstcalcenergy           = 100
nstenergy               = 1000
nstxout-compressed      = 1000
compressed-x-precision  = 1000
tcoupl                  = V-rescale
tc-grps                 = System
tau_t                   = 0.1
ref_t                   = {temperature}
pcoupl                  = Parrinello-Rahman
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = {pressure}
compressibility         = 4.5e-5
nstlist                 = 10
cutoff-scheme           = Verlet
ns_type                 = grid
coulombtype             = PME
rcoulomb                = 1.0
rvdw                    = 1.0
constraints             = h-bonds
constraint_algorithm    = lincs
lincs-iter              = 2
lincs-order             = 6
"""
    with open(workspace / f"{compound_id}_md.mdp", "w") as f:
        f.write(content)


def _fill_mdp_template(template_path: Path, **kwargs) -> str:
    """Read an .mdp template and fill {placeholder} tokens."""
    content = template_path.read_text()
    for key, value in kwargs.items():
        content = content.replace(f"{{{key}}}", str(value))
    return content
