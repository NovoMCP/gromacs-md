"""QM→FF bridge tool — wraps MCPB.py to convert QM output to AMBER topology.

Accepts a user's Gaussian or ORCA output file for a metalloprotein
QM calculation and produces ready-to-simulate AMBER force field
parameter files (.frcmod + .prep) plus a GROMACS-compatible topology.

The tool automates the "horrendous cutting and pasting between quantum
log files and classical force fields" that practitioners describe as
the main pain point in metalloprotein MD setup. See doc 12 for the
full execution plan and Nash's feedback that motivated this scope.

Architecture:
    1. Fragment extractor: PDB + MetalFinding → fragment .pdb around
       the metal coordination sphere
    2. MCPB.py input generator: MetalFinding → .in control file
    3. QM log handler: Gaussian .log/.fchk or ORCA .out/.hess → validate
       and prepare for MCPB.py
    4. MCPB.py runner: subprocess → collect .frcmod + .prep output
    5. Topology builder: tleap + parmed → GROMACS .top + .gro

Dependencies (all in AmberTools, already in Docker image):
    - MCPB.py
    - antechamber
    - parmchk2
    - tleap
    - parmed (pip, added in this session)
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# MCPB.py location — symlinked from AmberTools in the Dockerfile
MCPB_BIN = "MCPB.py"

# Coordination sphere cutoff (same as intake/parser.py uses)
COORD_CUTOFF_A = 2.8

# Known metal residue names in PDB files. Used to identify metal atoms
# regardless of whether they're ATOM or HETATM (pdb4amber sometimes
# reclassifies HETATM → ATOM for ions).
_METAL_RESNAMES = {
    "ZN", "MG", "CA", "FE", "MN", "CO", "NI", "CU", "CD", "HG",
    "NA", "K", "LI", "MO", "AG", "AU", "PT", "PD", "RU", "CR",
    "SR", "BA", "CS", "AL",
}


def _is_metal_line(line: str) -> bool:
    """Check if a PDB ATOM/HETATM line is a metal ion by residue name."""
    if not line.startswith(("ATOM  ", "HETATM")):
        return False
    try:
        resname = line[17:20].strip().upper()
        return resname in _METAL_RESNAMES
    except (IndexError, ValueError):
        return False


def prepare_pdb_for_mcpb(
    pdb_path: Path,
    output_dir: Path,
) -> Path:
    """Clean a raw RCSB PDB for MCPB.py using pdb4amber.

    Raw PDB files from RCSB have alternate conformations, non-standard
    residue names, missing hydrogens, and chain breaks that MCPB.py
    cannot handle. pdb4amber (AmberTools) is the standard cleanup tool
    used in every MCPB.py tutorial — it strips altlocs, renames waters,
    fixes atom names, and optionally adds hydrogens via reduce.

    This MUST run before MCPB.py step 1. Without it, gene_model_files
    crashes on the raw PDB structure.

    Returns: Path to the cleaned PDB file.
    """
    cleaned_pdb = output_dir / f"{pdb_path.stem}_amber.pdb"

    # Pre-extract metal HETATM lines BEFORE pdb4amber runs.
    # pdb4amber's --dry flag removes "solvent/ions" and some versions
    # treat single-atom HETATM ions (Zn2+, Mg2+, etc.) as solvent,
    # deleting them entirely. We preserve metal lines and re-inject
    # them after pdb4amber finishes.
    raw_lines = pdb_path.read_text(errors="replace").splitlines()
    metal_lines = [l for l in raw_lines if _is_metal_line(l)]
    if metal_lines:
        logger.info(
            f"Preserved {len(metal_lines)} metal line(s) before pdb4amber: "
            f"{[l[17:20].strip() for l in metal_lines[:5]]}"
        )

    # Resolve pdb4amber binary — ships with AmberTools but may live in
    # the micromamba env bin rather than on the system PATH.
    pdb4amber_bin = shutil.which("pdb4amber")
    if not pdb4amber_bin:
        for candidate in [
            "/opt/micromamba/envs/amber/bin/pdb4amber",
            "/opt/conda/envs/amber/bin/pdb4amber",
            "/usr/local/bin/pdb4amber",
        ]:
            if Path(candidate).exists():
                pdb4amber_bin = candidate
                break
    if not pdb4amber_bin:
        raise FileNotFoundError(
            "pdb4amber not found. Check that the AmberTools micromamba "
            "environment is installed and its bin/ is accessible."
        )

    cmd = [
        pdb4amber_bin,
        "-i", str(pdb_path),
        "-o", str(cleaned_pdb),
        "--dry",          # remove waters (MCPB.py adds its own)
        "--nohyd",        # don't add H yet (MCPB.py / tleap handles protonation)
        "--reduce",       # use reduce for atom name standardization
        "--keep-altlocs", # keep only the first altloc (A)
    ]

    # pdb4amber may not support all flags in every AmberTools version.
    # Fall back to a minimal invocation if the full command fails.
    logger.info(f"Running pdb4amber ({pdb4amber_bin}): {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(output_dir),
    )

    if result.returncode != 0:
        # Retry with minimal flags
        logger.warning(
            f"pdb4amber full command failed ({result.stderr[:200]}), "
            f"retrying with minimal flags"
        )
        cmd_minimal = [
            pdb4amber_bin,
            "-i", str(pdb_path),
            "-o", str(cleaned_pdb),
        ]
        result = subprocess.run(
            cmd_minimal,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(output_dir),
        )
        if result.returncode != 0:
            logger.error(f"pdb4amber failed: {result.stderr}")
            raise subprocess.CalledProcessError(
                result.returncode, "pdb4amber", stderr=result.stderr
            )

    if not cleaned_pdb.exists():
        raise FileNotFoundError(f"pdb4amber produced no output at {cleaned_pdb}")

    # Re-inject preserved metal lines if pdb4amber removed them.
    # Insert before the first END/TER that follows the last ATOM/HETATM.
    if metal_lines:
        cleaned_text = cleaned_pdb.read_text(errors="replace")
        cleaned_lines = cleaned_text.splitlines()

        # Check if metals survived pdb4amber
        metals_present = any(_is_metal_line(l) for l in cleaned_lines)
        if not metals_present:
            logger.warning(
                f"pdb4amber removed metal lines — re-injecting "
                f"{len(metal_lines)} preserved metal line(s)"
            )
            # Find the last ATOM/HETATM line and insert metals after it
            last_atom_idx = -1
            for i, line in enumerate(cleaned_lines):
                if line.startswith(("ATOM  ", "HETATM")):
                    last_atom_idx = i
            if last_atom_idx >= 0:
                # Renumber the metal atom serials to continue from the
                # last atom in the cleaned PDB
                last_serial = 0
                try:
                    last_serial = int(cleaned_lines[last_atom_idx][6:11].strip())
                except (IndexError, ValueError):
                    pass
                renumbered_metals = []
                for j, mline in enumerate(metal_lines):
                    new_serial = last_serial + j + 1
                    # PDB atom serial is columns 7-11 (right-justified)
                    padded = mline.ljust(80)
                    renumbered = f"{padded[:6]}{new_serial:5d}{padded[11:]}"
                    renumbered_metals.append(renumbered.rstrip())

                cleaned_lines = (
                    cleaned_lines[:last_atom_idx + 1]
                    + renumbered_metals
                    + cleaned_lines[last_atom_idx + 1:]
                )
                cleaned_pdb.write_text("\n".join(cleaned_lines) + "\n")
                logger.info(
                    f"Re-injected {len(renumbered_metals)} metal line(s), "
                    f"serials {last_serial+1}-{last_serial+len(renumbered_metals)}"
                )

    # pdb4amber sometimes strips the element column (columns 77-78) from
    # PDB ATOM/HETATM lines. MCPB.py's pymsmt parser uses this column to
    # determine the chemical element — without it, it falls back to
    # heuristics on the atom name that can misidentify "ZN" as nitrogen
    # instead of zinc. Fix the element column by deriving from atom name.
    _fix_element_column(cleaned_pdb)

    # AMBER force fields use protonation-state-specific histidine names:
    #   HID (delta-protonated), HIE (epsilon-protonated), HIP (doubly).
    # Raw PDB files use "HIS" (generic). pdb4amber with --nohyd can't
    # assign protonation states, so HIS remains unchanged. MCPB.py's
    # chargedict only knows HID/HIE/HIP — it crashes with KeyError: 'HIS'.
    # Default to HIE (epsilon-protonated, most common in proteins).
    # Also handle CYS → CYX for disulfide-bonded cysteines if needed.
    _AMBER_RESNAME_MAP = {
        "HIS": "HIE",   # epsilon-protonated histidine (most common default)
    }
    lines = cleaned_pdb.read_text(errors="replace").splitlines()
    renamed = 0
    fixed_lines = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 20:
            resname = line[17:20].strip()
            if resname in _AMBER_RESNAME_MAP:
                new_resname = _AMBER_RESNAME_MAP[resname]
                line = line[:17] + f"{new_resname:<3s}" + line[20:]
                renamed += 1
        fixed_lines.append(line)
    if renamed:
        cleaned_pdb.write_text("\n".join(fixed_lines) + "\n")
        logger.info(f"Renamed {renamed} residues for AMBER compatibility (HIS→HIE)")

    logger.info(f"PDB prepared for MCPB.py: {cleaned_pdb}")
    return cleaned_pdb


# Two-letter elements that could be confused with two single-letter atoms
_TWO_LETTER_ELEMENTS = {
    "ZN", "MG", "CA", "FE", "MN", "CO", "NI", "CU", "CD", "HG",
    "NA", "CL", "BR", "SE", "SI", "AL", "AS", "LI", "BE", "CR",
    "MO", "AG", "AU", "PT", "PD", "RU", "RH", "IR", "OS", "RE",
    "BI", "SN", "PB", "TI", "SR", "BA", "CE", "GA", "GE", "IN",
    "SB", "TE", "CS", "LA",
}


def _fix_element_column(pdb_path: Path) -> None:
    """Ensure columns 77-78 (element symbol) are populated in every
    ATOM/HETATM line. Derives the element from the atom name when the
    column is empty, using standard PDB naming conventions."""
    lines = pdb_path.read_text(errors="replace").splitlines()
    fixed = []
    changed = 0
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            # Pad to at least 80 chars
            padded = line.ljust(80)
            elem_col = padded[76:78].strip()
            if not elem_col:
                atom_name = padded[12:16].strip().upper()
                # Derive element from atom name
                if atom_name[:2] in _TWO_LETTER_ELEMENTS:
                    # Standard chemical notation: Zn, Mg, Fe (not ZN, MG, FE)
                    # MCPB.py's pymsmt parser reads "ZN" as element N (strips Z)
                    elem = atom_name[0].upper() + atom_name[1].lower()
                elif len(atom_name) >= 1 and atom_name[0].isalpha():
                    elem = atom_name[0].upper()
                else:
                    elem = atom_name.lstrip("0123456789")[:1].upper() or "X"
                padded = padded[:76] + f"{elem:>2s}" + padded[78:]
                changed += 1
            fixed.append(padded.rstrip())
        else:
            fixed.append(line)
    if changed:
        pdb_path.write_text("\n".join(fixed) + "\n")
        logger.info(f"Fixed element column for {changed} atoms in {pdb_path.name}")


def resolve_metal_resid(
    pdb_content: str,
    supplied_metal_resid: int,
    raw_pdb_content: Optional[str] = None,
) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Pre-flight resolver for the metal resid the caller supplied.

    audit_system reports the PDB-author residue numbering; parameterize_metal's
    `pdb4amber --dry --nohyd --reduce` pipeline renumbers residues during
    preprocessing. The two often disagree (observed 2026-06-06 on 1OKL: audit
    said 262, post-pdb4amber zinc lives at 257). The documented Nash chain
    "audit → use the resid in Phase 1" then 500s on the renumbering shift.

    Strategy ("Option C+" from QM-FF-BRIDGE-COMPLETION.md):

    - Scan the post-pdb4amber PDB for metal-residue lines.
    - If the supplied resid matches an actual metal, pass through unchanged.
    - **Single-metal proteins**: auto-resolve to the only metal.
    - **Multi-metal proteins** (e.g. 1OKL has HG + ZN): the supplied resid
      identifies which site the user wants — but we don't have its element
      after pdb4amber renumbered. If `raw_pdb_content` is provided, look up
      the supplied resid in the raw PDB to find its element. Then if
      exactly one metal of that element exists in the post-pdb4amber PDB,
      auto-resolve to it. This handles the 1OKL case: user passes 262, raw
      PDB has ZN@262, post-pdb4amber has HG@<x> + ZN@257 → resolves to 257
      because there's exactly one ZN.

    Returns: (effective_metal_resid, correction_or_None). The correction
    dict has shape:
      {"supplied": int, "used": int, "reason": str, "detected_metals": [str]}
    """
    detected = []  # list of (resid:int, element:str) tuples in POST-pdb4amber
    for line in pdb_content.splitlines():
        if not _is_metal_line(line):
            continue
        try:
            resid = int(line[22:26].strip())
            element = (line[76:78].strip() or line[12:16].strip()[:2]).upper()
            detected.append((resid, element))
        except (IndexError, ValueError):
            continue

    if not detected:
        # No metals found by our parser; let the existing downstream path
        # raise its enriched ValueError so the user sees the full diagnostic.
        return supplied_metal_resid, None

    # If the supplied resid already matches an actual metal, no correction.
    if any(r == supplied_metal_resid for r, _ in detected):
        return supplied_metal_resid, None

    # Single-metal case — auto-resolve unambiguously.
    if len(detected) == 1:
        actual_resid, actual_element = detected[0]
        return actual_resid, _build_correction_dict(
            supplied_metal_resid, actual_resid, detected,
            reason_suffix="this protein has exactly one metal",
        )

    # Multi-metal case — use the raw PDB to figure out which element the
    # user meant (their resid points at a specific element in the raw PDB).
    # Then look for exactly one metal of that element in post-pdb4amber.
    if raw_pdb_content:
        target_element = None
        for line in raw_pdb_content.splitlines():
            if not _is_metal_line(line):
                continue
            try:
                resid = int(line[22:26].strip())
                if resid == supplied_metal_resid:
                    target_element = (
                        line[76:78].strip() or line[12:16].strip()[:2]
                    ).upper()
                    break
            except (IndexError, ValueError):
                continue
        if target_element:
            same_element = [(r, e) for r, e in detected if e == target_element]
            if len(same_element) == 1:
                actual_resid, actual_element = same_element[0]
                return actual_resid, _build_correction_dict(
                    supplied_metal_resid, actual_resid, detected,
                    reason_suffix=(
                        f"multi-metal protein, but raw PDB resid "
                        f"{supplied_metal_resid} is {target_element} and the "
                        f"post-pdb4amber PDB has exactly one {target_element} site"
                    ),
                )

    # Multi-metal + ambiguous — can't disambiguate safely. Let downstream
    # raise the enriched ValueError so the caller can pick on retry.
    return supplied_metal_resid, None


def _build_correction_dict(supplied: int, used: int, detected: List[Tuple[int, str]], reason_suffix: str) -> Dict[str, Any]:
    """Helper to keep resolve_metal_resid focused on logic, not formatting."""
    logger.warning(
        "metal_resid mismatch — caller passed %d, auto-resolved to %d "
        "(%s). Option C+ from QM-FF-BRIDGE-COMPLETION.md.",
        supplied, used, reason_suffix,
    )
    return {
        "supplied": supplied,
        "used": used,
        "reason": (
            "Caller (e.g. audit_system) likely reported the PDB-author residue id; "
            "parameterize_metal needs the post-pdb4amber id, and these can disagree "
            f"on the same PDB. {reason_suffix.capitalize()}, so the parameterization "
            "is unambiguous — used the resolved resid and proceeded."
        ),
        "detected_metals": [f"{el}@{r}" for r, el in detected],
    }


def extract_chain_for_mcpb(
    pdb_path: Path,
    metal_resid: int,
    output_dir: Path,
) -> Path:
    """Extract the single chain containing the target metal for MCPB.py.

    MCPB.py works best on single-chain structures. Multi-chain PDBs
    (e.g., 1E67 with 4 Zn ions across 4 chains) confuse the model
    builder because it can't disambiguate which chain's metal is the
    target. The gene_model_files → build_large_model step crashes
    when multiple chains are present.

    Identifies which chain contains the metal at `metal_resid` and
    writes a single-chain PDB. If the metal isn't found or is in
    multiple chains, returns the original PDB unchanged.

    Returns: Path to the single-chain PDB (or the original if extraction
    wasn't needed/possible).
    """
    pdb_text = pdb_path.read_text(errors="replace")
    lines = pdb_text.splitlines()

    # Find which chain the target metal is in. Match by RESIDUE NAME
    # against known metals rather than record type — pdb4amber sometimes
    # reclassifies HETATM → ATOM for metal ions, and standard amino acid
    # ATOM records can share the same resid (e.g., ALA B 129 coexists
    # with ZN B 129).
    target_chain = None
    for line in lines:
        if not _is_metal_line(line):
            continue
        try:
            resid_str = line[22:26].strip()
            if resid_str == str(metal_resid):
                chain = line[21:22].strip() or "A"
                target_chain = chain
                break
        except (IndexError, ValueError):
            continue

    if not target_chain:
        logger.warning(
            f"Could not find metal at resid {metal_resid} to determine chain; "
            f"passing full PDB to MCPB.py"
        )
        return pdb_path

    # Check if PDB is already single-chain
    chains_seen = set()
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
            chains_seen.add(line[21:22].strip() or "A")
    if len(chains_seen) <= 1:
        logger.info(f"PDB is already single-chain ({target_chain})")
        return pdb_path

    # Extract only the target chain + its HETATMs
    logger.info(
        f"Multi-chain PDB ({len(chains_seen)} chains: {sorted(chains_seen)}). "
        f"Extracting chain {target_chain} (contains metal at resid {metal_resid})"
    )
    kept_lines = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            chain = line[21:22].strip() or "A"
            if chain == target_chain:
                kept_lines.append(line)
        elif line.startswith(("TER", "END", "HEADER", "TITLE", "CRYST", "REMARK")):
            kept_lines.append(line)

    if not any(l.startswith(("ATOM", "HETATM")) for l in kept_lines):
        logger.warning(f"Chain extraction produced no atoms; returning original PDB")
        return pdb_path

    single_chain_pdb = output_dir / f"{pdb_path.stem}_chain{target_chain}.pdb"
    single_chain_pdb.write_text("\n".join(kept_lines) + "\n")

    atom_count = sum(1 for l in kept_lines if l.startswith(("ATOM", "HETATM")))
    logger.info(f"Single-chain PDB: {atom_count} atoms from chain {target_chain}")
    return single_chain_pdb


def extract_metal_fragment(
    pdb_path: Path,
    metal_resid: int,
    output_dir: Path,
    cutoff: float = COORD_CUTOFF_A + 3.0,
) -> Path:
    """Extract a QM fragment from a PDB around a specific metal site.

    Keeps the metal atom and all residues with ANY atom within `cutoff`
    of the metal. The larger cutoff (5.8 Å by default) ensures we capture
    second-shell residues that may contribute to the electrostatic
    environment, which matters for RESP charge fitting.

    The classifier's MetalFinding.ligating_residues gives the first-shell
    donors, but the QM fragment needs to be larger (typically 50-150 atoms)
    to produce good force constants and charges.

    Returns: Path to the extracted fragment PDB.
    """
    try:
        import MDAnalysis as mda
    except ImportError:
        raise ImportError("MDAnalysis required for fragment extraction")

    u = mda.Universe(str(pdb_path))

    # Find the metal atom
    metal_atoms = u.select_atoms(f"resid {metal_resid}")
    if len(metal_atoms) == 0:
        # Build an actionable diagnostic. Common causes:
        #  - pdb4amber renumbering during preprocessing (HETATMs stripped,
        #    residues renumbered sequentially)
        #  - User supplied a different PDB than the one referenced in the
        #    QM log (cluster model vs full structure)
        #  - Residue is in a chain MDAnalysis parsed under a different ID
        # Surface the actual resids present in the PDB so the user can
        # correct or use audit_system to find the right resid.
        try:
            all_resids = sorted({int(r.resid) for r in u.residues})
            # Find HETATM/non-standard residues likely to be metals
            metal_candidates = []
            for r in u.residues:
                resname = (r.resname or "").strip()
                if resname in {"ZN", "FE", "CU", "MN", "NI", "CO", "CD",
                               "MG", "CA", "MO", "W", "V", "CR"}:
                    metal_candidates.append(f"{resname}@{r.resid}")
            # Truncate the resid list to keep the error message readable
            resid_summary = (
                f"{all_resids[:5]}...{all_resids[-5:]}" if len(all_resids) > 12
                else str(all_resids)
            )
            metal_hint = (
                f" Detected metal residues in this PDB: {', '.join(metal_candidates)}."
                if metal_candidates else
                f" No metal residues detected in this PDB."
            )
            raise ValueError(
                f"No atoms found at resid {metal_resid} in the supplied PDB.{metal_hint} "
                f"All resids present: {resid_summary}. "
                f"Common causes: (1) pdb4amber renumbered residues during preprocessing — "
                f"check the original PDB for the correct resid; "
                f"(2) the QM log was generated from a custom cluster that doesn't share "
                f"residue numbering with this PDB — use Phase 1's generated .com files "
                f"instead of a custom cluster; "
                f"(3) wrong chain — try audit_system on this PDB to find the metal site."
            )
        except ValueError:
            raise  # Re-raise our enriched ValueError
        except Exception:
            # If diagnostic generation itself fails, fall back to the
            # original error message rather than masking the real failure.
            raise ValueError(f"No atoms found at resid {metal_resid}")

    metal = metal_atoms[0]

    # Select all residues with any atom within cutoff of the metal
    near = u.select_atoms(f"around {cutoff} index {metal.index}")
    # Include the metal itself
    fragment = near.residues.atoms | metal_atoms

    # Write the fragment PDB
    fragment_pdb = output_dir / f"fragment_metal{metal_resid}.pdb"
    fragment.write(str(fragment_pdb))

    logger.info(
        f"Extracted QM fragment: {len(fragment)} atoms, "
        f"{len(fragment.residues)} residues around metal resid {metal_resid}"
    )
    return fragment_pdb


# Standard amino acids + common solvent/ion that don't need naa parameterization
_STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL",
    # Common protonation/naming variants
    "HID", "HIE", "HIP", "CYX", "CYM", "ASH", "GLH",
    # Waters and ions (stripped by pdb4amber --dry, but just in case)
    "HOH", "WAT", "SOL", "NA", "CL", "K", "MG", "CA", "ZN", "FE",
    "CU", "MN", "NI", "CO", "CD",
}


def parameterize_naa_residues(
    pdb_path: Path,
    metal_resid: int,
    metal_element: str,
    output_dir: Path,
    cutoff: float = COORD_CUTOFF_A + 1.0,
) -> tuple[List[Path], List[Path]]:
    """Detect and parameterize non-standard residues near the metal.

    MCPB.py requires mol2 + frcmod files for any non-amino-acid (NAA)
    residue in the metal coordination sphere. This is the same step
    that the MCPB.py tutorial handles manually:
        antechamber -fi pdb -fo mol2 -i LIGAND.pdb -o LIGAND.mol2 -c bcc
        parmchk2 -i LIGAND.mol2 -o LIGAND.frcmod -f mol2

    Scans the PDB for HETATM residues within `cutoff` of the metal
    that aren't standard amino acids, waters, or known ions. For each
    one, extracts it to a PDB, runs antechamber + parmchk2.

    Returns:
        (mol2_files, frcmod_files) — lists of paths for each NAA residue.
        Empty lists if no NAA residues are found near the metal.
    """
    pdb_text = pdb_path.read_text(errors="replace")
    lines = pdb_text.splitlines()

    # Find the metal atom coordinates
    metal_xyz = None
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            resid_str = line[22:26].strip()
            element = (line[76:78].strip() or line[12:16].strip()[:2]).upper()
            if resid_str == str(metal_resid) and element == metal_element.upper():
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                metal_xyz = (x, y, z)
                break
        except (IndexError, ValueError):
            continue

    if metal_xyz is None:
        logger.warning(
            "Could not find metal %s@%d in %s for NAA scan — returning empty "
            "naa_mol2/frcmod. MCPB.py will then fail with 'required in "
            "naa_mol2files but not provided' for any non-standard coordinating "
            "residue. Common cause: pdb4amber stripped the metal HETATM or "
            "rewrote the element column.",
            metal_element, metal_resid, pdb_path.name,
        )
        return [], []

    # Collect non-standard residues within cutoff of the metal.
    #
    # We accept BOTH `ATOM` and `HETATM` records here. pdb4amber sometimes
    # reclassifies non-standard residues between the two during its dry/reduce
    # passes; the HETATM-only filter previously used here missed those rows
    # (observed 2026-06-06 on 1OKL/MNS — MNS atoms made it through pdb4amber
    # but as ATOM records, so the scan returned empty and MCPB.py step 1
    # raised `MNS is required in naa_mol2files but not provided`). The
    # `_STANDARD_RESIDUES` set is the authoritative gate for what's standard:
    # amino acids (incl. protonation variants), waters, common ions. Anything
    # else IS a non-standard residue regardless of which record type it's in.
    naa_residues: Dict[str, List[str]] = {}  # resname → FULL atom lines (one instance)

    # Pass 1: group every non-standard atom by residue INSTANCE (chain, resid,
    # resname) and record each instance's closest approach to the metal.
    instance_atoms: Dict[tuple, List[str]] = {}
    instance_mindist: Dict[tuple, float] = {}
    scanned = 0
    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        resname = line[17:20].strip().upper()
        if resname in _STANDARD_RESIDUES or resname == metal_element.upper():
            continue  # standard residue / the metal itself
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except (IndexError, ValueError):
            continue
        scanned += 1
        key = (line[21], line[22:26].strip(), resname)  # chain, resid, resname
        instance_atoms.setdefault(key, []).append(line)
        dist = ((x - metal_xyz[0])**2 + (y - metal_xyz[1])**2 + (z - metal_xyz[2])**2) ** 0.5
        if key not in instance_mindist or dist < instance_mindist[key]:
            instance_mindist[key] = dist

    # Pass 2: a residue instance coordinates the metal if ANY of its atoms is
    # within cutoff — but we then parameterize the COMPLETE residue, not just
    # the in-cutoff atoms. The cutoff decides *whether* a residue coordinates,
    # never *which atoms* to parameterize: previously the atoms were collected
    # inside the `dist <= cutoff` test, so antechamber received only the 3/17
    # in-cutoff atoms of 1OKL's MNS (S, O1S, N3S) — a broken-valence fragment
    # that crashed sqm, leaving naa_mol2files empty and MCPB.py raising "MNS is
    # required in naa_mol2files but not provided". For each residue NAME we keep
    # the single closest instance (the one-copy template MCPB.py matches).
    candidates_seen: List[str] = []
    for key in sorted((k for k, d in instance_mindist.items() if d <= cutoff),
                      key=lambda k: instance_mindist[k]):
        _chain, resid, resname = key
        if resname in naa_residues:
            continue
        naa_residues[resname] = instance_atoms[key]
        candidates_seen.append(
            f"{resname}{resid}@{instance_mindist[key]:.2f}Å "
            f"({len(instance_atoms[key])} atoms)"
        )

    if not naa_residues:
        # Log diagnostic so future debugging can tell "no NSRs in PDB" from
        # "NSRs exist but all beyond cutoff".
        logger.info(
            "No non-standard residues found within %.1f Å of metal "
            "(scanned %d non-standard atom records total). If MCPB.py later "
            "complains about a missing naa_mol2files entry, the residue is "
            "beyond the cutoff — bump COORD_CUTOFF_A + N or invoke "
            "parameterize_naa_residues with a wider cutoff.",
            cutoff, scanned,
        )
        return [], []
    logger.info(
        "NSR coordination scan found %d residue(s) within %.1f Å of %s@%d: %s "
        "(parameterizing the complete residue, not just in-cutoff atoms)",
        len(naa_residues), cutoff, metal_element, metal_resid,
        ", ".join(candidates_seen),
    )

    mol2_files: List[Path] = []
    frcmod_files: List[Path] = []

    # Prefer the conda-env binaries directly. /usr/local/bin/antechamber is a
    # wrapper shell that tries to `source /usr/local/amber.sh` (which doesn't
    # exist in this image) — invoking it crashes with `amber.sh: No such file
    # or directory` even when the underlying tool is otherwise fine. The
    # /opt/micromamba/envs/amber/bin path skips the broken wrapper.
    _AMBER_BIN_PRIMARY = "/opt/micromamba/envs/amber/bin"
    antechamber_bin = (
        f"{_AMBER_BIN_PRIMARY}/antechamber"
        if Path(f"{_AMBER_BIN_PRIMARY}/antechamber").exists()
        else (shutil.which("antechamber") or "/usr/local/bin/antechamber")
    )
    parmchk2_bin = (
        f"{_AMBER_BIN_PRIMARY}/parmchk2"
        if Path(f"{_AMBER_BIN_PRIMARY}/parmchk2").exists()
        else (shutil.which("parmchk2") or "/usr/local/bin/parmchk2")
    )

    # AmberTools subprocesses need AMBERHOME for the auxiliary force-field
    # files antechamber + parmchk2 consult (parm, dat, leap/parm/*.frcmod).
    # When the wrapper script fails, AMBERHOME never gets set, and even the
    # underlying binary errors out on missing data files. Set it explicitly
    # so the subprocess environment is correct regardless of how we invoke
    # the tools.
    _AMBER_ENV = {
        **os.environ,
        "AMBERHOME": "/opt/micromamba/envs/amber",
        "PATH": f"{_AMBER_BIN_PRIMARY}:" + os.environ.get("PATH", ""),
    }

    for resname, res_lines in naa_residues.items():
        # Write the COMPLETE residue to its own PDB. antechamber preserves the
        # PDB atom names in the mol2, so MCPB.py can match the mol2 back to the
        # residue in original_pdb by name.
        naa_pdb = output_dir / f"{resname}.pdb"
        naa_pdb.write_text("\n".join(res_lines) + "\nEND\n")

        # antechamber → GAFF2 mol2. Try AM1-BCC (sqm) first, then fall back to
        # Gasteiger (-c gas, topological, no sqm). The post-pdb4amber ligand is
        # heavy-atoms-only (--nohyd; reduce only protonates standard residues),
        # so sqm — which needs explicit hydrogens — fails here; gas still yields
        # a complete mol2 whose atom count matches the H-less original_pdb, which
        # is what MCPB.py step 1 needs to build the residue template (metal-site
        # charges are refined from QM/RESP in the later phases). `-rn` forces the
        # residue name so MCPB.py's naa_mol2files lookup matches. The old `-c mul`
        # fallback was useless — Mulliken charges also run through sqm.
        naa_mol2 = output_dir / f"{resname}.mol2"
        for charge_method in ("bcc", "gas"):
            logger.info(f"Running antechamber for NAA {resname} (-c {charge_method})")
            result = subprocess.run(
                [antechamber_bin, "-fi", "pdb", "-fo", "mol2",
                 "-i", str(naa_pdb), "-o", str(naa_mol2),
                 "-c", charge_method, "-at", "gaff2", "-rn", resname, "-pf", "y"],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=str(output_dir),
                env=_AMBER_ENV,
            )
            if result.returncode == 0 and naa_mol2.exists():
                if charge_method == "gas":
                    logger.warning(
                        f"NAA {resname}: AM1-BCC unavailable (no hydrogens for "
                        f"sqm) — used Gasteiger charges. Charges are approximate; "
                        f"metal-site charges come from QM in the later phases."
                    )
                break
            logger.warning(
                f"antechamber -c {charge_method} failed for {resname}: "
                f"{(result.stderr or '')[:200]}"
            )
        if not naa_mol2.exists():
            logger.error(f"antechamber failed for {resname} (both bcc and gas)")
            continue

        mol2_files.append(naa_mol2)

        # parmchk2: mol2 → frcmod (fill missing parameters)
        naa_frcmod = output_dir / f"{resname}.frcmod"
        cmd_parmchk = [
            parmchk2_bin,
            "-i", str(naa_mol2),
            "-o", str(naa_frcmod),
            "-f", "mol2",
        ]
        logger.info(f"Running parmchk2 for NAA {resname}")
        result = subprocess.run(
            cmd_parmchk,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(output_dir),
            env=_AMBER_ENV,
        )
        if result.returncode != 0 or not naa_frcmod.exists():
            logger.warning(f"parmchk2 failed for {resname}: {result.stderr[:200]}")
            continue

        frcmod_files.append(naa_frcmod)
        logger.info(f"NAA {resname} parameterized: {naa_mol2.name}, {naa_frcmod.name}")

    return mol2_files, frcmod_files


def generate_mcpb_input(
    pdb_path: Path,
    metal_resid: int,
    metal_element: str,
    qm_software: str,
    charge: int,
    multiplicity: int,
    output_dir: Path,
    compound_id: str,
    ligating_residues: Optional[List[str]] = None,
    naa_mol2_files: Optional[List[Path]] = None,
    naa_frcmod_files: Optional[List[Path]] = None,
) -> Path:
    """Generate a MCPB.py input control file from classifier data.

    The control file tells MCPB.py where the metal is, what software
    produced the QM output, and what parameters to extract. The
    classifier's MetalFinding already has all the coordination sphere
    data needed.

    Args:
        pdb_path: Path to the full PDB (or fragment PDB).
        metal_resid: Residue number of the metal in the PDB.
        metal_element: Element symbol (e.g., "ZN", "MG", "CA").
        qm_software: "gaussian" or "orca".
        charge: Total charge of the QM fragment.
        multiplicity: Spin multiplicity.
        output_dir: Where to write the .in file.
        compound_id: Identifier for naming.
        ligating_residues: Optional list of "RESNAME:RESID:ATOM" from
            the classifier's MetalFinding. Used to validate the
            coordination sphere but not strictly required.

    Returns: Path to the generated .in file.
    """
    sw_map = {"gaussian": "g16", "orca": "orca"}
    sw_code = sw_map.get(qm_software.lower(), "g16")

    # MCPB.py's ion_ids expects the ATOM SERIAL NUMBER from the PDB
    # (the first column of ATOM/HETATM lines), NOT the residue sequence
    # number. Parse the PDB to find the serial number of the metal atom
    # at the given residue. If we can't find it, fall back to the
    # residue number and let MCPB.py's own error handling report it.
    ion_atom_id = metal_resid  # fallback
    pdb_text = pdb_path.read_text(errors="replace")
    for line in pdb_text.splitlines():
        if not _is_metal_line(line):
            continue
        try:
            resid_str = line[22:26].strip()
            if resid_str == str(metal_resid):
                ion_atom_id = int(line[6:11].strip())
                break
        except (IndexError, ValueError):
            continue
    logger.info(
        f"Metal {metal_element} at resid {metal_resid} → "
        f"atom serial {ion_atom_id}"
    )

    group_name = f"{metal_element}_{compound_id}"
    input_path = output_dir / f"{group_name}.in"

    # Build naa_mol2files and frcmod_files entries from parameterized NAA residues
    naa_mol2_str = " ".join(p.name for p in (naa_mol2_files or []))
    naa_frcmod_str = " ".join(p.name for p in (naa_frcmod_files or []))

    lines = [
        f"original_pdb {pdb_path.name}",
        f"group_name {group_name}",
        f"cut_off {COORD_CUTOFF_A}",
        f"ion_ids {ion_atom_id}",
        f"ion_mol2files {metal_element}.mol2",
        f"naa_mol2files {naa_mol2_str}".rstrip(),
        f"frcmod_files {naa_frcmod_str}".rstrip(),
        f"charge_model RESP",
        f"software_version {sw_code}",
        f"large_opt 0",  # Skip QM optimization — user already ran it
        f"force_field ff14SB",  # Base FF for non-metal residues
    ]

    input_path.write_text("\n".join(lines) + "\n")

    # Generate the ion mol2 file that MCPB.py expects in the working
    # directory. This is a standard monatomic-ion template — the same
    # file that ships with AmberTools at $AMBERHOME/dat/leap/parm/ but
    # may not be discoverable by MCPB.py when run from a temp workspace.
    # Generating it inline is safer than relying on AMBERHOME path resolution.
    _FORMAL_CHARGE = {
        "ZN": 2, "MG": 2, "CA": 2, "FE": 2, "FE3": 3, "MN": 2,
        "CO": 2, "NI": 2, "CU": 2, "CD": 2, "HG": 2, "NA": 1,
        "K": 1, "LI": 1,
    }
    fc = _FORMAL_CHARGE.get(metal_element.upper(), 2)
    mol2_path = output_dir / f"{metal_element.upper()}.mol2"
    # The mol2 atom type (column 6) determines how MCPB.py looks up the
    # element and its VDW parameters. MCPB.py's gauio.py parser derives
    # the element from the atom type using chemical element notation:
    #   "Zn" → element Zn (zinc)      ← CORRECT
    #   "ZN" → element N (nitrogen!)  ← WRONG (strips leading Z)
    #
    # The atom NAME (column 2) and residue name use uppercase (ZN), but
    # the atom TYPE must use standard element notation (Zn, Mg, Ca, Fe).
    # @<TRIPOS>SUBSTRUCTURE section is required by some MCPB.py code paths.
    me_upper = metal_element.upper()          # ZN — for residue/atom name
    me_title = metal_element.capitalize()     # Zn — for atom TYPE (element)
    mol2_content = (
        f"@<TRIPOS>MOLECULE\n"
        f"{me_upper}\n"
        f" 1 0 1 0 0\n"
        f"SMALL\n"
        f"NO_CHARGES\n"
        f"\n"
        f"\n"
        f"@<TRIPOS>ATOM\n"
        f"      1 {me_upper:<4s}        0.0000    0.0000    0.0000 {me_title:<2s}        1 {me_upper:<4s}    {fc:.4f}\n"
        f"@<TRIPOS>BOND\n"
        f"@<TRIPOS>SUBSTRUCTURE\n"
        f"     1 {me_upper:<4s}        1 TEMP              0 ****  ****    0 ROOT\n"
    )
    mol2_path.write_text(mol2_content)
    logger.info(f"Generated ion mol2: {mol2_path}")

    logger.info(f"Generated MCPB.py input: {input_path}")
    return input_path


def _run_mcpb_step(
    input_file: Path,
    working_dir: Path,
    step: int,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Run a single MCPB.py step and return the CompletedProcess."""
    mcpb_path = shutil.which(MCPB_BIN)
    if mcpb_path:
        cmd = [mcpb_path, "-i", input_file.name, "-s", str(step)]
    else:
        cmd = ["python3", "-m", "MCPB", "-i", input_file.name, "-s", str(step)]

    logger.info(f"Running MCPB.py step {step}: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(working_dir),
    )
    if result.returncode != 0:
        # Combine both streams — MCPB.py writes errors to both stdout and stderr
        combined = f"STDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        logger.error(f"MCPB.py step {step} failed:\n{combined}")
        raise subprocess.CalledProcessError(
            result.returncode, "MCPB.py", stderr=combined
        )
    return result


def run_mcpb_step1(
    input_file: Path,
    working_dir: Path,
) -> Dict[str, Path]:
    """MCPB.py step 1 — extract fragment + generate QM input files.

    This is Phase 1 of the two-phase parameterization workflow. It
    produces a Gaussian .com (or ORCA .inp) file that defines the
    exact fragment and atom ordering MCPB.py expects. The user must
    run QM on THIS file — not their own fragment extraction — because
    MCPB.py's step 3-4 requires atom-for-atom correspondence between
    the .com it generated and the .log it processes.

    Returns:
        Dict with output file paths: {"com_file": ..., "small_model_pdb": ...,
        "large_model_pdb": ..., ...}
    """
    _run_mcpb_step(input_file, working_dir, step=1)

    outputs: Dict[str, Path] = {}
    # MCPB.py step 1 produces Gaussian .com files for small and large models
    for pattern, key in [
        ("*_small_opt.com", "small_opt_com"),
        ("*_small_fc.com", "small_fc_com"),
        ("*_large_mk.com", "large_mk_com"),
        ("*_small_opt.pdb", "small_model_pdb"),
        ("*_large.pdb", "large_model_pdb"),
    ]:
        matches = list(working_dir.glob(pattern))
        if matches:
            outputs[key] = matches[0]

    # The .com files are what the user needs to run in Gaussian/ORCA.
    # Typically: small_fc.com (Hessian on the small model) and
    # large_mk.com (ESP charges on the large model). Some workflows
    # also need small_opt.com (geometry optimization first).
    logger.info(f"MCPB.py step 1 outputs: {list(outputs.keys())}")
    return outputs


def run_mcpb_step34(
    input_file: Path,
    working_dir: Path,
) -> Dict[str, Path]:
    """MCPB.py steps 3-4 — process QM output → produce FF parameters.

    This is Phase 2 of the two-phase workflow. The user's QM .log
    files (generated from the .com files produced by step 1) must be
    in the working directory with the names MCPB.py expects (same
    names as the .com files but with .log extension).

    Step 3: Extract force constants from Hessian, fit RESP charges
    Step 4: Generate .frcmod, .prep, tleap input → build topology

    Returns:
        Dict with output file paths: {"frcmod": ..., "prep": ...,
        "tleap_input": ..., "prmtop": ..., "inpcrd": ...}
    """
    _run_mcpb_step(input_file, working_dir, step=3)
    _run_mcpb_step(input_file, working_dir, step=4)

    outputs: Dict[str, Path] = {}
    for pattern, key in [
        ("*.frcmod", "frcmod"),
        ("*.prep", "prep"),
        ("*tleap*.in", "tleap_input"),
        ("*.prmtop", "prmtop"),
        ("*.inpcrd", "inpcrd"),
    ]:
        matches = list(working_dir.glob(pattern))
        if matches:
            outputs[key] = matches[0]

    logger.info(f"MCPB.py step 3-4 outputs: {list(outputs.keys())}")
    return outputs


# Keep the old run_mcpb as a convenience wrapper for direct API testing
def run_mcpb(
    input_file: Path,
    qm_log_path: Path,
    working_dir: Path,
    step: int = 4,
) -> Dict[str, Path]:
    """Legacy single-call MCPB.py runner. Prefer run_mcpb_step1 + run_mcpb_step34."""
    log_dest = working_dir / qm_log_path.name
    if not log_dest.exists() and qm_log_path.exists():
        shutil.copy2(qm_log_path, log_dest)
    _run_mcpb_step(input_file, working_dir, step=step)

    outputs: Dict[str, Path] = {}
    for pattern, key in [
        ("*.frcmod", "frcmod"),
        ("*.prep", "prep"),
        ("*tleap*.in", "tleap_input"),
        ("*.prmtop", "prmtop"),
        ("*.inpcrd", "inpcrd"),
    ]:
        matches = list(working_dir.glob(pattern))
        if matches:
            outputs[key] = matches[0]
    return outputs


def convert_mcpb_to_gromacs(
    prmtop: Path,
    inpcrd: Path,
    output_dir: Path,
    compound_id: str,
) -> tuple[Path, Path]:
    """Convert MCPB.py AMBER output to GROMACS format via parmed.

    Same conversion as membrane.convert_to_gromacs but for the
    metalloprotein parameterization output.
    """
    import parmed

    gro_file = output_dir / f"{compound_id}_metal.gro"
    top_file = output_dir / f"{compound_id}_metal.top"

    amber = parmed.load_file(str(prmtop), str(inpcrd))
    amber.save(str(top_file), overwrite=True)
    amber.save(str(gro_file), overwrite=True)

    logger.info(f"GROMACS topology from MCPB.py: {top_file}, {gro_file}")
    return gro_file, top_file


def validate_qm_log(qm_log_path: Path, qm_software: str, role: str = "") -> Dict:
    """Quick validation that the QM log file is parseable.

    Checks for key sections that MCPB.py will need:
    - Gaussian .log: Normal termination, Hessian (Force constants), ESP charges
    - Gaussian .fchk: Cartesian Force Constants, Total Energy, atomic numbers
    - ORCA .out: FINAL SINGLE POINT ENERGY, Hessian in .hess file

    `role` selects per-slot validation for the two-log Phase 2 workflow (the
    Hessian and the MK ESP come from two separate Gaussian runs — small_fc/freq
    and large_mk/Pop(MK) — so neither log has both sections):
    - role="hessian": require the Hessian (force constants). Used for the
      small_fc log. Does NOT require ESP.
    - role="esp": require MK/ESP charges. Used for the large_mk log. Does NOT
      require the Hessian.
    - role="" (default): legacy single-combined-log behavior (Hessian required).

    Returns a dict with {valid: bool, sections_found: list, warnings: list}.
    """
    content = qm_log_path.read_text(errors="replace")
    result: Dict = {"valid": False, "sections_found": [], "warnings": []}
    filename = qm_log_path.name.lower()

    if qm_software.lower() == "gaussian":
        # Detect format: .fchk uses different section headers than .log
        is_fchk = filename.endswith(".fchk") or "Number of atoms" in content[:500]

        if is_fchk:
            # Formatted checkpoint file — preferred format for MCPB.py
            # Contains Hessian in clean matrix layout, optimized geometry,
            # and wavefunction. No "Normal termination" marker.
            checks = [
                ("Cartesian Force Constants", "hessian"),
                ("Number of atoms", "atoms"),
                ("Total Energy", "energy"),
                ("Current cartance coordinates", "geometry"),  # typo is in the real format
            ]
            for pattern, name in checks:
                if pattern in content:
                    result["sections_found"].append(name)
            # Also check for the actual coordinate data
            if "Current cartesian coordinates" in content:
                result["sections_found"].append("geometry")
            # fchk is valid if it has the Hessian + atom count
            result["valid"] = "hessian" in result["sections_found"]
            if not result["valid"]:
                result["warnings"].append(
                    "No Hessian found in .fchk — file may be from an optimization "
                    "without freq. Re-run with Freq keyword to include force constants."
                )
            result["sections_found"].append("fchk_format")
        else:
            # Standard .log format
            checks = [
                ("Normal termination", "normal_termination"),
                ("Force constants in Cartesian coordinates", "hessian"),
                ("ESP charges", "esp_charges"),
                ("Merz-Kollman", "mk_charges"),
            ]
            for pattern, name in checks:
                if pattern in content:
                    result["sections_found"].append(name)

            sections = result["sections_found"]
            terminated = "normal_termination" in sections
            has_hessian = "hessian" in sections
            has_esp = "esp_charges" in sections or "mk_charges" in sections

            if not terminated:
                result["warnings"].append(
                    "Gaussian job did not terminate normally — results may be incomplete"
                )

            if role == "esp":
                # large_mk slot: needs MK ESP charges, NOT the Hessian.
                if not has_esp:
                    result["warnings"].append(
                        "esp_log is missing Pop(MK) ESP charges — re-run the "
                        "large_mk job with Pop(MK,ReadRadii)."
                    )
                result["valid"] = terminated and has_esp
            else:
                # role == "hessian" (small_fc slot) or legacy default: needs the Hessian.
                if not has_hessian:
                    msg = (
                        "hessian_log is missing the Hessian — re-run the small_fc "
                        "job with the freq keyword."
                        if role == "hessian"
                        else "No Hessian found in Gaussian output — force constants "
                        "cannot be extracted. Re-run with freq or force keyword."
                    )
                    result["warnings"].append(msg)
                result["valid"] = terminated and has_hessian

    elif qm_software.lower() == "orca":
        checks = [
            ("FINAL SINGLE POINT ENERGY", "energy"),
            ("MULLIKEN ATOMIC CHARGES", "charges"),
            ("TOTAL RUN TIME", "completed"),
        ]
        for pattern, name in checks:
            if pattern in content:
                result["sections_found"].append(name)

        # ORCA Hessian is in a separate .hess file
        hess_path = qm_log_path.with_suffix(".hess")
        if hess_path.exists():
            result["sections_found"].append("hessian_file")
        else:
            result["warnings"].append(
                f"ORCA .hess file not found at {hess_path.name} — "
                f"force constants cannot be extracted. Re-run with Freq keyword."
            )

        result["valid"] = (
            "completed" in result["sections_found"]
            and ("hessian_file" in result["sections_found"]
                 or "hessian" in result["sections_found"])
        )

    else:
        result["warnings"].append(f"Unknown QM software: {qm_software}")

    # Summarize what's missing for actionable error messages
    if not result["valid"]:
        if qm_software.lower() == "gaussian":
            expected = {"normal_termination", "hessian", "esp_charges"}
            if any(s == "fchk_format" for s in result["sections_found"]):
                expected = {"hessian", "atoms"}
            found = set(result["sections_found"])
            missing = expected - found
            if missing:
                result["missing_sections"] = sorted(missing)
                result["fix_hint"] = (
                    f"Missing: {', '.join(sorted(missing))}. "
                    f"Ensure the Gaussian job includes 'Freq Pop=MK' keywords "
                    f"and completes normally."
                )
        elif qm_software.lower() == "orca":
            expected = {"energy", "completed", "hessian_file"}
            found = set(result["sections_found"])
            missing = expected - found
            if missing:
                result["missing_sections"] = sorted(missing)
                result["fix_hint"] = (
                    f"Missing: {', '.join(sorted(missing))}. "
                    f"Ensure the ORCA job includes 'Freq' keyword, completes "
                    f"normally, and the .hess file is uploaded alongside the .out."
                )

    return result
