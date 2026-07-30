"""POST /audit — classify a structure without running MD."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..state import AppState, get_state
from ..auth import validate_api_key
from ..models import AuditRequest
from ..pdb_handling import fetch_pdb_from_rcsb
from intake import classify_structure

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/audit", dependencies=[Depends(validate_api_key)])
async def audit_system(request: AuditRequest, state: AppState = Depends(get_state)):
    """Classify a structure without running MD — synchronous system audit.

    Runs the same intake classifier used by /simulate, but returns the
    full RoutingDecision directly instead of creating a job. Useful for:
      - Pre-flight qualification before submitting an MD job
      - Free "System Audit" tool — inspect any PDB for membrane status,
        metal sites, heme, Fe-S clusters, PTMs, etc.
      - Batch screening to identify which targets the current pipeline
        can handle vs. which need future branches (charmm36m_membrane,
        mcpb_distal, qmmm_active_site)

    Returns within ~5 seconds. No GPU, no Redis job record, no credits
    consumed downstream. Equivalent classifier path to the /simulate
    intake gate — if this returns run_soluble, an MD submission for the
    same structure will pass the gate; if it returns refused, the
    submission will refuse with the same reasons and suggested_branch.
    """
    if not request.pdb_id and not request.pdb_content:
        raise HTTPException(
            status_code=400,
            detail="Provide either pdb_id or pdb_content",
        )

    pdb_content = request.pdb_content
    pdb_id = request.pdb_id

    # Fetch from RCSB if only pdb_id was given — matches /simulate behavior.
    if pdb_id and not pdb_content:
        try:
            pdb_content = await fetch_pdb_from_rcsb(pdb_id)
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch {pdb_id} from RCSB: {e}",
            )

    try:
        decision = await classify_structure(
            pdb_content=pdb_content,
            pdb_id=pdb_id,
            redis_client=state.redis,
        )
    except Exception as e:
        logger.exception(f"Classifier raised on audit request: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classifier error: {e}",
        )

    # Return the full decision as JSON. Pydantic handles serialization.
    return {
        "status": "audited",
        "would_route_to": decision.route,
        "suggested_branch": decision.suggested_branch,
        "primary_reason": decision.reasons[0] if decision.reasons else None,
        "reasons": decision.reasons,
        "profile": decision.profile.model_dump(),
    }
