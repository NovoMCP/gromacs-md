"""Membrane system builder — bilayer construction around an oriented protein.

Uses packmol-memgen (AmberTools) to build a lipid bilayer around a
membrane protein, then optionally converts the AMBER-format output to
GROMACS-compatible topology via parmed. Designed to be called from the
membrane branch of run_simulation_pipeline.

v1 scope (doc 11):
- Protein-only membrane MD (no ligand in the bilayer yet)
- Pure POPC bilayer (symmetric, single lipid type)
- packmol-memgen handles orientation internally when --preoriented
  is not set (auto-detect transmembrane region)
- CHARMM36m for protein + lipids, GAFF2 for ligand (unchanged)

Validation criteria before promoting the route:
- Area per lipid: 65-68 Å² for POPC
- Electron density profile symmetry across leaflets
- Deuterium SCD order parameters (0.15-0.25 for palmitoyl chain)
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default lipid composition for v1 (pure POPC).
# v1.1 adds POPC/cholesterol mixtures; v2 adds asymmetric leaflets.
DEFAULT_LIPID = "POPC"
DEFAULT_LIPID_RATIO = 1.0

# Buffer distance (Å) between the protein and the box edge in the
# membrane plane (XY). Along Z (membrane normal), packmol-memgen
# adds water above and below the bilayer automatically.
DEFAULT_DIST = 15.0

# Salt concentration (M) for physiological ionic strength.
DEFAULT_SALT_CONC = 0.15


def build_membrane_system(
    protein_pdb: Path,
    output_dir: Path,
    compound_id: str,
    *,
    lipid: str = DEFAULT_LIPID,
    lipid_ratio: float = DEFAULT_LIPID_RATIO,
    dist: float = DEFAULT_DIST,
    salt_conc: float = DEFAULT_SALT_CONC,
    preoriented: bool = False,
) -> tuple[Path, Path, dict]:
    """Build a lipid bilayer around a membrane protein via packmol-memgen.

    Args:
        protein_pdb: Path to the input protein PDB file. If `preoriented`
            is False (default), packmol-memgen auto-detects the
            transmembrane region and orients the protein.
        output_dir: Working directory for intermediate and output files.
        compound_id: Identifier used for naming output files.
        lipid: Lipid type (default "POPC"). Must be in packmol-memgen's
            lipid library.
        lipid_ratio: Mole fraction of this lipid (1.0 for pure bilayer).
        dist: Buffer distance in Å between protein and box edge in XY.
        salt_conc: NaCl concentration in M.
        preoriented: If True, skip auto-orientation and trust the input
            PDB's coordinate frame as membrane-aligned (Z = normal).

    Returns:
        (system_pdb, system_prmtop, build_stats) where:
            system_pdb: Path to the solvated system PDB (AMBER format)
            system_prmtop: Path to the AMBER topology (.prmtop)
            build_stats: Dict with lipid counts, box dimensions, etc.

    Raises:
        subprocess.CalledProcessError if packmol-memgen fails.
        FileNotFoundError if packmol-memgen is not on PATH.
    """
    system_prefix = output_dir / f"{compound_id}_membrane"

    cmd = [
        "packmol-memgen",
        "--pdb", str(protein_pdb),
        "--lipids", f"{lipid}:{lipid_ratio}",
        "--salt",
        "--salt_conc", str(salt_conc),
        "--dist", str(dist),
        "--output", str(system_prefix),
        "--overwrite",
    ]
    if preoriented:
        cmd.append("--preoriented")

    logger.info(f"Running packmol-memgen: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min max — large systems can be slow
        cwd=str(output_dir),
    )

    if result.returncode != 0:
        logger.error(f"packmol-memgen failed:\n{result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, "packmol-memgen", stderr=result.stderr
        )

    # packmol-memgen outputs: {prefix}.pdb and {prefix}.prmtop
    system_pdb = Path(f"{system_prefix}.pdb")
    system_prmtop = Path(f"{system_prefix}.prmtop")

    if not system_pdb.exists():
        # Some versions of packmol-memgen use slightly different naming
        candidates = list(output_dir.glob(f"{compound_id}_membrane*.[pP][dD][bB]"))
        if candidates:
            system_pdb = candidates[0]
        else:
            raise FileNotFoundError(
                f"packmol-memgen produced no output PDB at {system_pdb}"
            )

    # Collect build stats from output
    build_stats = _parse_memgen_output(result.stdout)
    build_stats["lipid"] = lipid
    build_stats["salt_conc"] = salt_conc
    build_stats["preoriented"] = preoriented

    logger.info(
        f"Membrane system built: {build_stats.get('total_atoms', '?')} atoms, "
        f"{build_stats.get('n_lipids', '?')} lipids"
    )

    return system_pdb, system_prmtop, build_stats


def convert_to_gromacs(
    prmtop: Path,
    inpcrd_or_pdb: Path,
    output_dir: Path,
    compound_id: str,
) -> tuple[Path, Path]:
    """Convert AMBER topology to GROMACS format via parmed.

    Args:
        prmtop: Path to AMBER .prmtop topology file.
        inpcrd_or_pdb: Path to AMBER .inpcrd or .pdb coordinates.
        output_dir: Where to write GROMACS output files.
        compound_id: Identifier for naming output files.

    Returns:
        (gro_file, top_file): Paths to the GROMACS .gro and .top files.
    """
    import parmed

    gro_file = output_dir / f"{compound_id}_membrane.gro"
    top_file = output_dir / f"{compound_id}_membrane.top"

    logger.info(f"Converting AMBER topology to GROMACS format via parmed")
    amber = parmed.load_file(str(prmtop), str(inpcrd_or_pdb))
    amber.save(str(top_file), overwrite=True)
    amber.save(str(gro_file), overwrite=True)

    logger.info(f"GROMACS topology written: {top_file}, {gro_file}")
    return gro_file, top_file


def _parse_memgen_output(stdout: str) -> dict:
    """Extract stats from packmol-memgen stdout.

    Best-effort parsing — returns whatever we can find without failing
    if the output format changes between AmberTools versions.
    """
    stats: dict = {}
    for line in stdout.splitlines():
        lower = line.lower().strip()
        if "total number of atoms" in lower:
            try:
                stats["total_atoms"] = int(lower.split()[-1])
            except (ValueError, IndexError):
                pass
        if "number of lipid" in lower or "lipid molecules" in lower:
            try:
                stats["n_lipids"] = int(
                    "".join(c for c in lower.split(":")[-1] if c.isdigit())
                )
            except (ValueError, IndexError):
                pass
        if "box dimensions" in lower or "box size" in lower:
            stats["box_info"] = line.strip()
    return stats
