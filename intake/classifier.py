"""Intake classifier orchestration — the production entry point.

Ties together parser, opm, metalpdb, pfam, and heuristics to produce
a RoutingDecision for a given PDB. This is what `main.py` calls.

v0 routing (Option A, conservative):
    - Soluble protein, no metals, no membrane, no heme/Fe-S → run_soluble
    - Everything else → refused, with a specific reason and a
      `suggested_branch` hint pointing at the future branch that would
      handle the case when built.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import List, Optional

import httpx

from . import metalpdb as mpdb
from . import opm
from . import parser
from .cofactors import magnesium_sites
from .covalent import covalent_branch_refusal, covalent_ligand_link_records
from .heuristics import classify_by_donors
from .modified_residues import unsupported_modified_residues
from .models import (
    ClassificationSource,
    FunctionalRole,
    MetalFinding,
    RoutingDecision,
    SuggestedBranch,
    SystemProfile,
)
from .pfam import lookup_role

logger = logging.getLogger(__name__)

# Membrane proteins (OPM) run in a POPC bilayer; MEMBRANE_BRANCH_ENABLED=0 refuses them.
MEMBRANE_BRANCH_ENABLED = os.environ.get("MEMBRANE_BRANCH_ENABLED", "1") == "1"
# Covalent ligands the covalent branch supports (one single-residue ligand on
# a Cys SG) run there; COVALENT_BRANCH_ENABLED=0 refuses every covalent ligand.
COVALENT_BRANCH_ENABLED = os.environ.get("COVALENT_BRANCH_ENABLED", "1") == "1"


async def classify_structure(
    *,
    pdb_content: str,
    pdb_id: Optional[str] = None,
    redis_client=None,
) -> RoutingDecision:
    """Classify a PDB and return a routing decision.

    Args:
        pdb_content: The PDB file as a string. Required.
        pdb_id: Optional 4-char PDB ID. When present, enables OPM and
            MetalPDB lookups. When absent (user upload without matching
            ID), classification falls back to local + heuristic only.
        redis_client: Optional async Redis client for API response
            caching. When None, every call hits the external APIs.

    Returns:
        A RoutingDecision. Never raises for classification concerns —
        parser failures produce `route="refused"` with a specific reason.
        Only hard infrastructure failures (all downstream dependencies
        completely broken) would raise, which should be treated as
        job failures by the caller.
    """
    profile = SystemProfile(pdb_id=pdb_id)
    reasons: List[str] = []
    covalent = False  # a supported covalent ligand: run_covalent unless refused below

    # 1. Parse the structure locally (CPU-bound — off the event loop).
    loop = asyncio.get_event_loop()
    parsed = await loop.run_in_executor(None, parser.parse_pdb_content, pdb_content)

    profile.parser_used = parsed.parser_used  # type: ignore[assignment]
    profile.warnings.extend(parsed.warnings)
    profile.hetatm_inventory = parsed.hetatm_inventory
    profile.heme_residues = parsed.heme_residues
    profile.fes_clusters = parsed.fes_clusters

    if parsed.parser_used == "failed":
        reasons.append("structure could not be parsed")
        return RoutingDecision(route="refused", reasons=reasons, profile=profile)

    # 2. Membrane detection via OPM (only when pdb_id is available).
    async with httpx.AsyncClient() as http_client:
        if pdb_id:
            is_membrane, opm_src, opm_warn = await opm.check_membrane(
                pdb_id, http_client=http_client, redis_client=redis_client
            )
            profile.is_membrane = is_membrane
            profile.opm_source = opm_src
            if opm_warn:
                profile.warnings.append(opm_warn)

        # 3. HETATM audit refusals — heme and Fe-S clusters bail out
        # before the MetalPDB call because we refuse them unconditionally
        # in v1 per doc 08.
        if profile.heme_residues:
            reasons.append(
                f"heme cofactor detected ({profile.heme_residues}); "
                f"refuse in v1 (requires specialized parameterization)"
            )
            return RoutingDecision(
                route="refused", reasons=reasons, profile=profile, suggested_branch=None
            )

        if profile.fes_clusters:
            summary = ", ".join(f"{n}x{k}" for k, n in sorted(profile.fes_clusters.items()))
            reasons.append(
                f"iron-sulfur cluster detected ({summary}); "
                f"refuse in v1 pending ML/MM branch"
            )
            return RoutingDecision(
                route="refused", reasons=reasons, profile=profile, suggested_branch=None
            )

        modified = unsupported_modified_residues(pdb_content)
        profile.ptm_residues = [r.label() for r in modified]
        if modified:
            shown = "; ".join(profile.ptm_residues[:4])
            more = f" and {len(modified) - 4} more" if len(modified) > 4 else ""
            reasons.append(
                f"modified residues in the protein chain ({shown}{more}); the force "
                f"field has no parameters for them, and dropping them would join the "
                f"neighbouring residues across the gap. The structure is refused with "
                f"any ligand; use one without these modifications"
            )
            return RoutingDecision(
                route="refused", reasons=reasons, profile=profile, suggested_branch="modified_residues"
            )

        links = covalent_ligand_link_records(pdb_content)
        profile.covalent_links = [link.label() for link in links]
        if links:
            unsupported = (covalent_branch_refusal(pdb_content, links) if COVALENT_BRANCH_ENABLED
                           else "the covalent branch is turned off")
            if unsupported:
                reasons.append(
                    f"covalent ligand bonded to the protein ({'; '.join(profile.covalent_links[:3])}): "
                    f"{unsupported}. The soluble branch would simulate it as a non-bonded ligand, "
                    f"so the structure is refused with any ligand; for a non-covalent study use a "
                    f"structure without the covalent bond (apo, or with a reversible inhibitor)"
                )
                return RoutingDecision(
                    route="refused", reasons=reasons, profile=profile, suggested_branch="covalent_ligand"
                )
            covalent = True
            reasons.append(
                f"covalent ligand bonded to the protein ({profile.covalent_links[0]}); routing to "
                f"the covalent branch (ligand and cysteine as one residue, ff14SB + GAFF2)"
            )

        if profile.is_membrane:
            if MEMBRANE_BRANCH_ENABLED:
                reasons.append(
                    "membrane protein (found in OPM); routing to the membrane "
                    "branch (packmol-memgen bilayer + semi-isotropic NPT)"
                )
            else:
                reasons.append(
                    "membrane protein (found in OPM); the membrane branch "
                    "(lipid bilayer) is not available yet"
                )
            # Still report metal content below if any, for full audit visibility.

        # 4. MetalPDB lookup (only when pdb_id is available and we have metals).
        annotations = []
        if parsed.metal_shells and pdb_id:
            annotations, mpdb_warn = await mpdb.fetch_sites(
                pdb_id, http_client=http_client, redis_client=redis_client
            )
            if mpdb_warn:
                profile.warnings.append(mpdb_warn)

    # 5. Per-metal classification + routing contribution.
    consumed: set = set()
    has_catalytic = False
    has_structural_or_unknown = False
    # Mg2+ on a nucleotide phosphate with an all-oxygen shell is simulated as a cofactor
    kept_mg = {(s.chain, int(s.resseq)): s for s in magnesium_sites(pdb_content)
               if s.supported and s.resseq.lstrip("-").isdigit()}

    for shell in parsed.metal_shells:
        ann = mpdb.match_annotation(
            shell.metal_element, shell.metal_chain, shell.metal_resid,
            annotations, consumed,
        )
        role, source, reason = _classify_metal(shell, ann)

        geometry_sanity_ok: Optional[bool] = None
        if ann is not None and ann.coord_number is not None:
            geometry_sanity_ok = abs(ann.coord_number - shell.coord_number) <= 1
            if not geometry_sanity_ok:
                profile.warnings.append(
                    f"coordination mismatch on {shell.metal_element}@{shell.metal_resid}: "
                    f"local={shell.coord_number}, MetalPDB={ann.coord_number} — review"
                )

        finding = MetalFinding(
            element=shell.metal_element,
            chain=shell.metal_chain or None,
            residue_number=shell.metal_resid,
            coordination_number=shell.coord_number,
            ligating_residues=shell.ligating_residues,
            fingerprint=shell.fingerprint(),
            metalpdb_site_id=ann.site_id if ann else None,
            metalpdb_pfam=ann.pfam if ann else None,
            metalpdb_ec=ann.ec_number if ann else None,
            metalpdb_nuclearity=ann.nuclearity if ann else None,
            metalpdb_geometry=ann.geometry if ann else None,
            metalpdb_pattern=ann.pattern if ann else None,
            metalpdb_coord_number=ann.coord_number if ann else None,
            metalpdb_molecule=ann.molecule if ann else None,
            functional_role=role,
            classification_source=source,
            classification_reason=reason,
            geometry_sanity_ok=geometry_sanity_ok,
        )
        profile.metal_sites.append(finding)

        label = (
            f"{shell.metal_element}:{shell.metal_chain}@{shell.metal_resid}"
            if shell.metal_chain
            else f"{shell.metal_element}@{shell.metal_resid}"
        )
        kept = kept_mg.get((shell.metal_chain, shell.metal_resid)) if shell.metal_element == "MG" else None
        if kept is not None:
            res, chain, seq = kept.nucleotide
            profile.kept_cofactors.append(f"{kept.label()} with {res} {chain}{seq}")
            reasons.append(
                f"metal {label}: nucleotide-bound Mg2+ with {res}, kept as a cofactor "
                f"(Mg2+ as a nonbonded 12-6 ion with its {len(kept.waters)} crystal waters, "
                f"{res} with GAFF2 AM1-BCC charges)")
            continue
        reasons.append(f"metal {label}: {role} ({reason})")

        if role == "catalytic":
            has_catalytic = True
        elif role in ("structural", "electron", "transport", "unknown"):
            has_structural_or_unknown = True

    # 6. Final routing.
    # Membrane proteins run in the membrane branch (MEMBRANE_BRANCH_ENABLED=0 refuses them).
    # Metals and other non-trivial features refuse.
    if profile.is_membrane:
        if MEMBRANE_BRANCH_ENABLED and not covalent:
            return RoutingDecision(
                route="run_membrane", reasons=reasons, profile=profile, suggested_branch=None,
            )
        return RoutingDecision(
            route="refused", reasons=reasons, profile=profile, suggested_branch="charmm36m_membrane",
        )

    if has_catalytic:
        return RoutingDecision(
            route="refused",
            reasons=reasons,
            profile=profile,
            suggested_branch="qmmm_active_site",
        )

    if has_structural_or_unknown:
        return RoutingDecision(
            route="refused",
            reasons=reasons,
            profile=profile,
            suggested_branch="mcpb_distal",
        )

    if covalent:
        return RoutingDecision(route="run_covalent", reasons=reasons, profile=profile, suggested_branch=None)

    # Happy path — no metals, no membrane, no exotic cofactors.
    reasons.append("no metals, no membrane, no exotic cofactors — happy path")
    return RoutingDecision(
        route="run_soluble", reasons=reasons, profile=profile, suggested_branch=None
    )


def _classify_metal(
    shell: parser.CoordinationShell,
    annotation: Optional[mpdb.MetalPDBAnnotation],
) -> tuple[FunctionalRole, ClassificationSource, str]:
    """Reason from (annotation + local shell) to a functional role.

    Order:
        1. Pfam lookup (strongest signal when MetalPDB has a hit)
        2. EC number presence (enzyme → default catalytic)
        3. Coordination pattern heuristic (e.g. HX(1)HX(22)H → catalytic)
        4. Donor-atom heuristic table (fallback)
    """
    if annotation is not None:
        # 1. Pfam
        pfam_role = lookup_role(annotation.pfam)
        if pfam_role is not None:
            return pfam_role, "metalpdb_pfam", f"Pfam {annotation.pfam!r} -> {pfam_role}"

        # 2. EC presence
        if annotation.ec_number:
            return (
                "catalytic",
                "metalpdb_ec",
                f"EC {annotation.ec_number} (enzyme) and Pfam {annotation.pfam!r} "
                f"not in lookup — default catalytic",
            )

        # 3. Pattern heuristic
        pattern = (annotation.pattern or "").upper()
        if pattern:
            if pattern.count("H") >= 3 and "C" not in pattern:
                return (
                    "catalytic",
                    "metalpdb_pattern",
                    f"pattern {pattern!r} (3xHis) -> catalytic",
                )
            if pattern.count("C") >= 2 and pattern.count("H") >= 2:
                return (
                    "structural",
                    "metalpdb_pattern",
                    f"pattern {pattern!r} (C2H2-like) -> structural",
                )

    # 4. Donor-atom heuristic fallback (no annotation, or annotation
    # present but no signal we can interpret).
    role, reason = classify_by_donors(shell.metal_element, shell.ligating_residues)
    return role, "heuristic", reason
