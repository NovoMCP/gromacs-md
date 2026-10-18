"""Keep the job result small enough for an MCP client to read in full."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from .config import RESULT_SERIES_MAX_POINTS

logger = logging.getLogger(__name__)

# Summaries first, per-frame data last: a client that truncates the result
# then loses graph points, not the findings.
SECTION_ORDER = [
    "quality_report", "mmgbsa", "ligand_dynamics", "quality_metrics",
    "rmsd", "rmsf", "radius_of_gyration", "checkpoint", "log_diagnostics",
    "equilibration",
]


def thin_series(points: List, max_points: int) -> List:
    """Evenly spaced subset of `points`, always keeping the first and last."""
    n = len(points)
    if n <= max_points or max_points < 2:
        return points
    step = (n - 1) / (max_points - 1)
    return [points[round(i * step)] for i in range(max_points)]


def _thin(obj: Any, max_points: int) -> Any:
    if isinstance(obj, list):
        return [_thin(v, max_points) for v in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    for key, value in obj.items():
        if key == "trajectory" and isinstance(value, list) and len(value) > max_points:
            out[key] = thin_series(value, max_points)
            out["trajectory_points_full"] = len(value)
        else:
            out[key] = _thin(value, max_points)
    return out


def compact_analysis(analysis: Dict, max_points: int = RESULT_SERIES_MAX_POINTS) -> Dict:
    """Thinned, summary-first copy of `analysis`; the input is not modified."""
    thinned = _thin(analysis, max_points)
    ordered = {k: thinned[k] for k in SECTION_ORDER if k in thinned}
    ordered.update((k, v) for k, v in thinned.items() if k not in ordered)
    return ordered


def write_full_analysis(workspace: Path, compound_id: str, analysis: Dict) -> str:
    """Save the full-resolution analysis next to the trajectory; returns the file name."""
    name = f"{compound_id}_analysis_full.json"
    try:
        (Path(workspace) / name).write_text(json.dumps(analysis))
    except OSError as e:
        logger.warning(f"Could not write {name}: {e}")
        return ""
    return name
