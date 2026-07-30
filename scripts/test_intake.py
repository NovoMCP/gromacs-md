#!/usr/bin/env python3
"""Live integration test for the intake classifier.

Runs classify_structure against three real PDB IDs using the live
RCSB, OPM, and MetalPDB APIs. Produces human-readable output and the
full structured RoutingDecision JSON for each case.

Intended as a pre-merge sanity check — see the unified diff review
for the intake/ package port. Not wired into CI.

Usage:
    python scripts/test_intake.py

Expected outcomes (per planning/hard-systems/09-classifier-v0-validation.md
and the Nash distal/active-site decision rule):

    1UBQ  Ubiquitin              run_soluble        (no metals, no membrane)
    2RH1  β2 adrenergic receptor refused            charmm36m_membrane
    1CA2  Carbonic anhydrase II  refused            qmmm_active_site
                                                    (catalytic Zn, Pfam=Carb_anhydrase)
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Tuple

import httpx

from intake import classify_structure, RoutingDecision


RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

TEST_CASES: list[Tuple[str, str, str, str]] = [
    (
        "1UBQ",
        "Ubiquitin",
        "run_soluble",
        "no metals, no membrane — happy path",
    ),
    (
        "2RH1",
        "β2 adrenergic receptor",
        "refused + charmm36m_membrane",
        "OPM hit — membrane protein",
    ),
    (
        "1CA2",
        "Carbonic anhydrase II",
        "refused + qmmm_active_site",
        "MetalPDB → Pfam=Carb_anhydrase + EC 4.2.1.1 → catalytic Zn",
    ),
]


async def fetch_pdb(pdb_id: str, client: httpx.AsyncClient) -> str:
    """Download a PDB from RCSB. Raises on non-200."""
    resp = await client.get(RCSB_URL.format(pdb_id=pdb_id), timeout=30.0)
    resp.raise_for_status()
    return resp.text


def print_header(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def print_decision(pdb_id: str, decision: RoutingDecision) -> None:
    """Pretty-print the decision with reasons + structured JSON."""
    print(f"\n  Route:             {decision.route}")
    print(f"  Suggested branch:  {decision.suggested_branch}")
    print(f"  Is membrane:       {decision.profile.is_membrane}")
    print(f"  Parser used:       {decision.profile.parser_used}")
    print(f"  Metal sites:       {len(decision.profile.metal_sites)}")
    print(f"  Heme residues:     {decision.profile.heme_residues}")
    print(f"  Fe-S clusters:     {decision.profile.fes_clusters}")
    if decision.profile.warnings:
        print(f"  Warnings:          {decision.profile.warnings}")

    print("\n  Reasons:")
    for r in decision.reasons:
        print(f"    - {r}")

    if decision.profile.metal_sites:
        print("\n  Metal findings:")
        for m in decision.profile.metal_sites:
            chain_str = f":{m.chain}" if m.chain else ""
            print(
                f"    - {m.element}{chain_str}@{m.residue_number}  "
                f"{m.fingerprint}"
            )
            print(
                f"        role={m.functional_role}  "
                f"source={m.classification_source}  "
                f"pfam={m.metalpdb_pfam!r}  ec={m.metalpdb_ec!r}"
            )
            if m.classification_reason:
                print(f"        reason: {m.classification_reason}")
            if m.metalpdb_geometry:
                print(
                    f"        MetalPDB geom={m.metalpdb_geometry!r}  "
                    f"nuclearity={m.metalpdb_nuclearity!r}  "
                    f"cn_api={m.metalpdb_coord_number}"
                )
            if m.geometry_sanity_ok is not None:
                print(
                    f"        CheckMyMetal sanity: "
                    f"{'PASS' if m.geometry_sanity_ok else 'MISMATCH'}"
                )


def print_full_json(pdb_id: str, decision: RoutingDecision) -> None:
    """Dump the full RoutingDecision as it would be serialized to Redis."""
    print(f"\n  ── Full RoutingDecision JSON for {pdb_id} ──")
    print(decision.model_dump_json(indent=2))


async def main() -> int:
    print_header("INTAKE CLASSIFIER — LIVE API INTEGRATION TEST")
    print("  APIs: RCSB (structure) + OPM (membrane) + MetalPDB (annotation)")
    print("  Redis client: None (heuristic+live path, no cache)")

    results: list[Tuple[str, str, RoutingDecision]] = []

    async with httpx.AsyncClient() as fetch_client:
        for pdb_id, name, expected, description in TEST_CASES:
            print_header(f"{pdb_id} — {name}")
            print(f"  Expected:  {expected}")
            print(f"  Rationale: {description}")
            print(f"\n  Fetching {pdb_id} from RCSB…")
            try:
                pdb_content = await fetch_pdb(pdb_id, fetch_client)
                print(f"  Fetched {len(pdb_content)} bytes")
            except Exception as e:
                print(f"  FETCH FAILED: {e}")
                return 1

            print(f"  Classifying (hitting OPM + MetalPDB live)…")
            try:
                decision = await classify_structure(
                    pdb_content=pdb_content,
                    pdb_id=pdb_id,
                    redis_client=None,
                )
            except Exception as e:
                print(f"  CLASSIFIER RAISED: {type(e).__name__}: {e}")
                return 1

            print_decision(pdb_id, decision)
            print_full_json(pdb_id, decision)
            results.append((pdb_id, expected, decision))

    # --------------------------------------------------------------
    # Synthetic MetalPDB-miss validation
    # --------------------------------------------------------------
    # Closes the one untested code path from doc 09: when MetalPDB has
    # no annotation at all (AlphaFold models, user uploads, unreleased
    # PDB entries), the classifier must fall through to the Pfam-free
    # donor-atom heuristic and still produce a sensible routing
    # decision. Monkey-patches intake.metalpdb.fetch_sites to return
    # empty and re-runs 1ZNF + 1CA2 through classify_structure.
    print_header("SYNTHETIC METALPDB-MISS TEST (heuristic fallback path)")
    print("  Monkey-patching intake.metalpdb.fetch_sites to return empty.")
    print("  Expected: both PDBs still refuse via donor-atom heuristic.")

    from intake import metalpdb as _mpdb

    original_fetch = _mpdb.fetch_sites

    async def _empty_fetch(pdb_id, *, http_client, redis_client=None):
        """Force-miss stub — pretends MetalPDB has no data for any PDB."""
        return [], None

    _mpdb.fetch_sites = _empty_fetch  # type: ignore[assignment]
    miss_results: list[Tuple[str, str, RoutingDecision]] = []
    try:
        async with httpx.AsyncClient() as fetch_client:
            for pdb_id, name, description in [
                ("1ZNF", "Zinc finger", "Zn Cys2His2 heuristic (should route structural → refused)"),
                ("1CA2", "Carbonic anhydrase II", "Zn 3xHis heuristic (should route catalytic → refused)"),
            ]:
                print_header(f"{pdb_id} — {name} (MetalPDB forced miss)")
                print(f"  Rationale: {description}")
                print(f"\n  Fetching {pdb_id} from RCSB…")
                try:
                    pdb_content = await fetch_pdb(pdb_id, fetch_client)
                except Exception as e:
                    print(f"  FETCH FAILED: {e}")
                    return 1
                print(f"  Fetched {len(pdb_content)} bytes")
                print(f"  Classifying with stubbed MetalPDB (force-miss)…")
                decision = await classify_structure(
                    pdb_content=pdb_content,
                    pdb_id=pdb_id,
                    redis_client=None,
                )
                print_decision(pdb_id, decision)
                # Verify the classification source is "heuristic" — if
                # it's "metalpdb_pfam" something, the monkey-patch failed.
                for m in decision.profile.metal_sites:
                    if m.classification_source != "heuristic":
                        print(
                            f"  ::warning:: expected classification_source=heuristic, "
                            f"got {m.classification_source!r} on {m.element}@{m.residue_number}"
                        )
                miss_results.append((pdb_id, "heuristic fallback → refused", decision))
    finally:
        _mpdb.fetch_sites = original_fetch  # restore

    # Final scoreboard (combined live + synthetic-miss)
    print_header("SUMMARY")
    print(f"  {'PDB':<6} {'Route':<15} {'Suggested Branch':<22} Expected")
    print(f"  {'-'*6} {'-'*15} {'-'*22} {'-'*40}")
    for pdb_id, expected, decision in results:
        print(
            f"  {pdb_id:<6} {decision.route:<15} "
            f"{str(decision.suggested_branch):<22} {expected}"
        )
    print(f"  {'-'*6} {'-'*15} {'-'*22} {'-'*40}")
    print("  (synthetic MetalPDB miss — heuristic fallback path)")
    for pdb_id, expected, decision in miss_results:
        print(
            f"  {pdb_id:<6} {decision.route:<15} "
            f"{str(decision.suggested_branch):<22} {expected}"
        )
    print()

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
