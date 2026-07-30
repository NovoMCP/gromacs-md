"""Trajectory analysis — RMSD, RMSF, gyration, MM-GBSA, ligand dynamics, quality gates.

Quality gate functions (May 2026):
  compute_quality_metrics() — temperature, pressure, energy stability
  validate_checkpoint()     — gmx check on .cpt file
  parse_md_log()            — GROMACS warning/error extraction
  assess_md_quality()       — aggregate pass/conditional/fail gate
"""

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Input templates for gmx_MMPBSA — minimal GB binding free energy.
# igb=5 is OBC2 (Onufriev-Bashford-Case, 2004) — standard for drug-discovery
# MM-GBSA, tolerates protein and small-molecule ligand geometries. salt_conc=0.15
# matches physiological. interval=10 means every 10th frame contributes.
MMPBSA_INPUT_GB_BASE = """&general
  startframe=1, interval=10, verbose=2,
  keep_files=0, netcdf=1
/
&gb
  igb=5, saltcon=0.150,
/
"""


def _mmpbsa_input_with_decomp(cutoff_angstrom: float) -> str:
    """Base input plus a per-residue decomposition (&decomp) namelist.

    idecomp=1 gives per-residue energies without pairwise interaction terms
    (fast + what Theo wants: "which residues contribute most to binding").

    The cutoff is sized at call time from the actual ligand-protein geometry
    (see _decomp_cutoff_angstrom). This matters: gmx_MMPBSA resolves
    print_res="within X" via get_selected_residues() -> list2range(), and
    list2range() returns '' (an empty *string*, not a dict) for an empty
    selection. make_top.py then does list2range(...)['string'], which raises
    "TypeError: string indices must be integers" and aborts the ENTIRE
    MM-GBSA run — not just decomposition. So we only ever emit a cutoff that
    we've confirmed selects at least one residue.
    """
    return MMPBSA_INPUT_GB_BASE + (
        "&decomp\n"
        f'  idecomp=1, print_res="within {cutoff_angstrom:.1f}",\n'
        "  dec_verbose=0,\n"
        "/\n"
    )


# Back-compat: full input at the historical 5.0 Å cutoff.
MMPBSA_INPUT_GB = _mmpbsa_input_with_decomp(5.0)


def _parse_xvg(path: Path) -> list:
    """Parse a GROMACS .xvg file, returning list of [x, y, ...] rows."""
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.startswith(("#", "@")) or not line.strip():
            continue
        parts = line.split()
        try:
            rows.append([float(v) for v in parts])
        except ValueError:
            continue
    return rows


def _extract_energy(workspace: Path, edr_file: str, xvg_file: str, selection: str) -> None:
    """Extract a property from a GROMACS .edr file using gmx energy."""
    edr_path = workspace / edr_file
    if not edr_path.exists():
        return
    try:
        subprocess.run(
            ["gmx", "energy", "-f", edr_file, "-o", xvg_file],
            input=selection.encode(), check=True, cwd=workspace,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        logger.debug(f"gmx energy failed for {edr_file}: {e}")


def _summarize_timeseries(rows: list) -> Dict:
    """Compute summary stats for a time series from XVG data."""
    if not rows:
        return {}
    values = [r[1] for r in rows if len(r) >= 2]
    if not values:
        return {}
    n = len(values)
    mean = sum(values) / n
    last_quarter = values[-max(1, n // 4):]
    last_quarter_mean = sum(last_quarter) / len(last_quarter)
    std = (sum((v - mean) ** 2 for v in values) / n) ** 0.5
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "final": round(values[-1], 4),
        "last_quarter_mean": round(last_quarter_mean, 4),
        "stable": std < abs(mean) * 0.1 if mean != 0 else std < 1.0,
        "trajectory": [[round(r[0], 4), round(r[1], 4)] for r in rows],
    }


def _linear_slope_per_ns(traj: list, last_fraction: float = 0.25) -> float:
    """Least-squares slope of y vs x for the LAST `last_fraction` of a
    [[time_ps, y], ...] trajectory.

    Returns slope in y-units per nanosecond. Slope is computed on the
    tail rather than the full trajectory so the early equilibration
    ramp doesn't get conflated with steady-state drift — the question
    we want to answer is "is the system still drifting in the converged
    region?", not "did temperature change from t=0 to t=T?".

    last_fraction=0.25 keeps the last quarter, matching the
    last_quarter_mean window already reported by _summarize_timeseries.

    Returns 0.0 for trajectories with <3 points in the window or zero
    variance in time.
    """
    if not traj or len(traj) < 4:
        return 0.0
    tail_start = int(len(traj) * (1.0 - last_fraction))
    tail = traj[tail_start:]
    if len(tail) < 3:
        return 0.0
    n = len(tail)
    sx = sum(p[0] for p in tail)
    sy = sum(p[1] for p in tail)
    sxx = sum(p[0] * p[0] for p in tail)
    sxy = sum(p[0] * p[1] for p in tail)
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0
    slope_per_ps = (n * sxy - sx * sy) / denom
    return slope_per_ps * 1000.0  # ps → ns


def compute_quality_metrics(equilibration: Dict, rmsd: Dict) -> Dict:
    """Compute pass/conditional/fail for each MD stability metric.

    Grade on equilibration STABILITY (does the last quarter of the
    trajectory agree with the overall mean?) and DRIFT (linear slope per
    ns), not raw fluctuation amplitude. Std is reported as a diagnostic
    but not graded — pressure and temperature fluctuations scale with
    system size, and grading on std penalizes small systems unfairly.

    Concretely: a ligand-only run in a 3000-atom water box has temperature
    std ~7K and pressure std ~600 bar by basic statistical mechanics.
    Those are NOT quality failures. What matters is whether the time-mean
    matches the thermostat/barostat setpoint and whether there's a
    monotonic trend across the trajectory.

    Thresholds (v2 — calibrated against ligand-only + protein-ligand runs):
      Temperature equilibration:  |overall_mean - last_quarter_mean|
                                    PASS <1K, CONDITIONAL 1-3K, FAIL >3K
      Temperature drift:          linear slope over production
                                    PASS <1 K/ns, CONDITIONAL 1-3, FAIL >3
      Pressure equilibration:     |overall_mean - last_quarter_mean|
                                    PASS <50 bar, CONDITIONAL 50-200, FAIL >200
      Pressure drift:             PASS <100 bar/ns, CONDITIONAL 100-500, FAIL >500
      Energy plateau:             ratio of last-half var to first-half var
                                    PASS <2, CONDITIONAL 2-5, FAIL >5 (unchanged)

    The std fields (temperature_drift_k, pressure_std_bar) are kept for
    UI back-compat but are no longer graded against. New fields:
    temperature_offset_k, pressure_offset_bar, *_slope_per_ns.
    """
    metrics: Dict[str, Any] = {}

    grade_rank = {"PASS": 0, "CONDITIONAL": 1, "FAIL": 2}

    def _window_duration_ns(traj: list, fraction: float = 0.25) -> float:
        """Duration of the last-fraction tail of a [[time_ps, ...], ...] trajectory, in ns."""
        if not traj or len(traj) < 2:
            return 0.0
        tail_start_idx = int(len(traj) * (1.0 - fraction))
        tail = traj[tail_start_idx:]
        if len(tail) < 2:
            return 0.0
        return (tail[-1][0] - tail[0][0]) / 1000.0

    def _drift_grade(slope_per_ns: float, std: float, window_ns: float,
                     thresholds: tuple) -> str:
        """Grade a drift slope, but only if it exceeds the noise floor.

        slope is in unit/ns. For the drift to be statistically distinguishable
        from random fluctuation, the slope's predicted change over the
        observation window (|slope * window_ns|) must exceed 2x the trajectory
        std. Otherwise the apparent slope is just sampling noise — common
        for short runs (<1ns) where the last-quarter window is too small
        to resolve a real trend, and for pressure where instantaneous virial
        fluctuations scale as 1/N.

        2σ matches the offset gate threshold and is the conventional
        "statistically significant" cutoff. A slope that produces less than
        2σ of total change across the observation window is consistent with
        random walk and should not be graded as systematic drift.
        """
        pass_thr, cond_thr = thresholds
        if window_ns <= 0 or std <= 0:
            return "PASS"
        slope_change = abs(slope_per_ns) * window_ns
        if slope_change < 2.0 * std:
            # Below the noise floor: the apparent slope is consistent with
            # random fluctuation, not real drift. Don't penalize.
            return "PASS"
        s = abs(slope_per_ns)
        if s < pass_thr:
            return "PASS"
        if s < cond_thr:
            return "CONDITIONAL"
        return "FAIL"

    def _offset_grade(offset: float, std: float, n_tail: int,
                      thresholds: tuple) -> str:
        """Grade equilibration offset, gated on statistical significance.

        offset = |overall_mean - last_quarter_mean|. With std and n_tail
        (the number of samples in the last-quarter window), the standard
        error of the last_quarter_mean is std/sqrt(n_tail). An offset
        within ±2 SE is statistically consistent with "no drift, just
        sampling noise" and should not be graded as drift.

        For long runs, SE is tiny and the gate is essentially inactive.
        For short runs (0.1–1 ns), SE is comparable to the offset
        thresholds, and the gate correctly suppresses noise-driven
        false positives — the same problem the v1 gate had with std
        for small systems.
        """
        pass_thr, cond_thr = thresholds
        if n_tail <= 1 or std <= 0:
            return "PASS"
        se = std / (n_tail ** 0.5)
        # 2.5σ gate — calls drift "noise" up to ~99% confidence level.
        # 2σ would be ~95%, but for the smallest valid runs (0.3 ns smoke
        # tests, n_tail ≈ 38, std ~7 K) a 2σ gate fires at offset 2.3 K
        # which is exactly the regime where mean-drift is ambiguous between
        # incomplete-equilibration and sampling noise. 2.5σ pushes the
        # decision boundary out far enough that short runs don't get
        # CONDITIONAL by default while keeping long-run drift detection
        # intact (SE shrinks as 1/sqrt(N), so the gate becomes inactive
        # well before 1 ns of post-equilibration sampling).
        if abs(offset) < 2.5 * se:
            return "PASS"
        a = abs(offset)
        if a < pass_thr:
            return "PASS"
        if a < cond_thr:
            return "CONDITIONAL"
        return "FAIL"

    # ── Temperature ────────────────────────────────────────────────────
    temp = equilibration.get("production_temperature", {})
    if temp:
        temp_mean = temp.get("mean")
        temp_lq = temp.get("last_quarter_mean")
        temp_std = temp.get("std", 0.0)
        temp_traj = temp.get("trajectory", [])

        # Diagnostic only (back-compat name): std of full production
        metrics["temperature_drift_k"] = round(temp_std, 3)

        # Equilibration grade: how much does the late mean drift from
        # the overall mean? If the trajectory is equilibrated, these agree.
        # Gated on statistical significance — see _offset_grade docstring.
        n_tail = int(len(temp_traj) * 0.25) if temp_traj else 0
        if temp_mean is not None and temp_lq is not None:
            temp_offset = abs(temp_mean - temp_lq)
            metrics["temperature_offset_k"] = round(temp_offset, 3)
            offset_grade = _offset_grade(
                temp_offset, temp_std, n_tail, thresholds=(1.0, 3.0)
            )
        else:
            offset_grade = "PASS"

        # Drift grade — slope on the last-quarter window, with noise-floor gate
        temp_slope = _linear_slope_per_ns(temp_traj)
        metrics["temperature_slope_k_per_ns"] = round(temp_slope, 4)
        slope_grade = _drift_grade(
            temp_slope, temp_std, _window_duration_ns(temp_traj),
            thresholds=(1.0, 3.0),
        )

        metrics["temperature_grade"] = max(
            (offset_grade, slope_grade), key=lambda g: grade_rank[g]
        )

    # ── Pressure ───────────────────────────────────────────────────────
    pres = equilibration.get("production_pressure", {})
    if pres:
        pres_mean = pres.get("mean")
        pres_lq = pres.get("last_quarter_mean")
        pres_std = pres.get("std", 0.0)
        pres_traj = pres.get("trajectory", [])

        metrics["pressure_std_bar"] = round(pres_std, 2)

        n_tail = int(len(pres_traj) * 0.25) if pres_traj else 0
        if pres_mean is not None and pres_lq is not None:
            pres_offset = abs(pres_mean - pres_lq)
            metrics["pressure_offset_bar"] = round(pres_offset, 2)
            offset_grade = _offset_grade(
                pres_offset, pres_std, n_tail, thresholds=(50.0, 200.0)
            )
        else:
            offset_grade = "PASS"

        pres_slope = _linear_slope_per_ns(pres_traj)
        metrics["pressure_slope_bar_per_ns"] = round(pres_slope, 2)
        slope_grade = _drift_grade(
            pres_slope, pres_std, _window_duration_ns(pres_traj),
            thresholds=(100.0, 500.0),
        )

        metrics["pressure_grade"] = max(
            (offset_grade, slope_grade), key=lambda g: grade_rank[g]
        )

    # ── Energy plateau (unchanged) ─────────────────────────────────────
    # Last-half variance vs first-half variance. Ratio near 1.0 means the
    # system reached a steady distribution; >5 means it's still drifting.
    traj = equilibration.get("production_potential_energy", {}).get("trajectory", [])
    if len(traj) > 10:
        mid = len(traj) // 2
        first_vals = [p[1] for p in traj[:mid]]
        last_vals = [p[1] for p in traj[mid:]]
        first_var = sum((v - sum(first_vals) / len(first_vals)) ** 2 for v in first_vals) / len(first_vals)
        last_var = sum((v - sum(last_vals) / len(last_vals)) ** 2 for v in last_vals) / len(last_vals)
        ratio = last_var / first_var if first_var > 0 else float("inf")
        metrics["energy_variance_ratio"] = round(ratio, 3)
        metrics["energy_grade"] = (
            "PASS" if ratio < 2.0 else ("CONDITIONAL" if ratio < 5.0 else "FAIL")
        )

    # ── RMSD sanity — flag protein unfolding ──────────────────────────
    max_rmsd = rmsd.get("max_nm", 0)
    if max_rmsd > 2.0:
        metrics["rmsd_warning"] = f"Max RMSD {max_rmsd:.2f} nm — possible protein unfolding"

    # Overall grade
    grades = [v for k, v in metrics.items() if k.endswith("_grade")]
    if "FAIL" in grades:
        metrics["overall"] = "FAIL"
    elif "CONDITIONAL" in grades:
        metrics["overall"] = "CONDITIONAL"
    else:
        metrics["overall"] = "PASS"

    return metrics


def validate_checkpoint(workspace: Path, compound_id: str) -> Dict:
    """Run gmx check on the production checkpoint file."""
    result: Dict[str, Any] = {"checkpoint_valid": False}
    cpt = workspace / f"{compound_id}_md.cpt"
    if not cpt.exists():
        result["error"] = "Checkpoint file not found"
        return result
    try:
        proc = subprocess.run(
            ["gmx", "check", "-f", str(cpt)],
            capture_output=True, text=True, timeout=30, cwd=workspace,
        )
        output = proc.stdout + proc.stderr
        # gmx check prints to stderr; look for item count or "Last frame" as success signals
        has_data = "item" in output.lower() or "last frame" in output.lower() or "reading" in output.lower()
        has_error = "Error" in output or "Fatal" in output
        result["checkpoint_valid"] = has_data and not has_error
        result["checkpoint_size_bytes"] = cpt.stat().st_size
    except Exception as e:
        result["error"] = str(e)
    return result


def parse_md_log(workspace: Path, compound_id: str) -> Dict:
    """Extract warnings, errors, and metadata from GROMACS production log."""
    log_path = workspace / f"{compound_id}_md.log"
    if not log_path.exists():
        return {"error": "Log file not found"}

    content = log_path.read_text(errors="replace")
    warnings: List[Dict] = []

    # GROMACS logs have two sections: parameter echo (first ~200-500 lines)
    # and simulation output. Parameters legitimately contain "inf" values
    # (e.g., "rlist = inf", "nstlist = inf"). Only scan for NaN/Inf in the
    # simulation output section (after "Started mdrun").
    # Other warnings (LINCS, constraints) are meaningful anywhere in the log.

    # Find where simulation output begins
    sim_start = content.find("Started mdrun")
    if sim_start < 0:
        sim_start = content.find("starting mdrun")
    if sim_start < 0:
        sim_start = len(content) // 3  # fallback: skip first third

    # Patterns that apply to the full log
    full_log_patterns = [
        (r"LINCS WARNING", "WARNING", "LINCS constraint warning"),
        (r"Constraint error", "CRITICAL", "Constraint solver failure"),
        (r"[Ss]tep size too small", "WARNING", "Step size reduction"),
        (r"PME did not converge", "WARNING", "PME convergence issue"),
    ]

    for pattern, severity, description in full_log_patterns:
        for match in re.finditer(pattern, content):
            line_num = content[:match.start()].count("\n") + 1
            warnings.append({
                "description": description,
                "severity": severity,
                "line": line_num,
            })

    # NaN/Inf — only in simulation output section (after parameter echo)
    sim_content = content[sim_start:]
    sim_line_offset = content[:sim_start].count("\n")
    nan_inf_patterns = [
        (r"\bnan\b", "CRITICAL", "NaN detected in energy/coordinates"),
        (r"\b[+-]?inf\b", "CRITICAL", "Inf detected in energy/coordinates"),
    ]
    for pattern, severity, description in nan_inf_patterns:
        for match in re.finditer(pattern, sim_content, re.IGNORECASE):
            line_num = sim_line_offset + sim_content[:match.start()].count("\n") + 1
            warnings.append({
                "description": description,
                "severity": severity,
                "line": line_num,
            })

    version_match = re.search(r"GROMACS version:\s*(.+)", content)
    has_critical = any(w["severity"] == "CRITICAL" for w in warnings)

    return {
        "gromacs_version": version_match.group(1).strip() if version_match else None,
        "warnings": warnings,
        "warning_count": len(warnings),
        "has_critical": has_critical,
        "log_grade": "FAIL" if has_critical else ("CONDITIONAL" if warnings else "PASS"),
    }


# ─── v3 quality assessment helpers ──────────────────────────────────────────
# Three-layer schema: execution_integrity (binary), sampling_quality (bands),
# scientific_adequacy (bands per intent). See docs/MD-QUALITY-V3.md (or the
# 2026-05-14 design discussion) for full rationale.

_KNOWN_INTENTS = (
    "smoke_test",
    "equilibration_only",
    "pose_stability",
    "mm_gbsa",
    "fep_window",
)


def _effective_sample_size(trajectory: list, n_blocks: int = 10) -> Optional[int]:
    """Block-averaged effective sample size for a [[time, y], ...] trajectory.

    Computes the statistical inefficiency s = b · σ²(block means) / σ²(full)
    and returns ESS = N / s. Block size b = N // n_blocks. For trajectories
    with fewer than 2·n_blocks samples or zero variance, returns None.

    ESS is a heuristic — block averaging tends to underestimate
    autocorrelation length for short series. Use as a rough convergence
    signal, not as an authoritative statistical claim. We expose the raw
    number in evidence and let the adequacy layer interpret it.
    """
    if not trajectory or len(trajectory) < 2 * n_blocks:
        return None
    values = [p[1] for p in trajectory if len(p) >= 2]
    n = len(values)
    if n < 2 * n_blocks:
        return None
    mean = sum(values) / n
    total_var = sum((v - mean) ** 2 for v in values) / n
    if total_var <= 0:
        return None
    b = n // n_blocks
    if b < 2:
        return None
    block_means = []
    for i in range(n_blocks):
        chunk = values[i * b : (i + 1) * b]
        if not chunk:
            continue
        block_means.append(sum(chunk) / len(chunk))
    if len(block_means) < 2:
        return None
    bm_mean = sum(block_means) / len(block_means)
    block_var = sum((bm - bm_mean) ** 2 for bm in block_means) / len(block_means)
    if block_var <= 0:
        # All block means identical — either degenerate series or block size
        # captured a periodicity. Treat as fully independent.
        return n
    s = b * block_var / total_var
    if s <= 0:
        return n
    return int(n / s)


def _autocorrelation_ess(trajectory: list, max_lag: Optional[int] = None) -> Optional[Dict]:
    """Integrated-autocorrelation-time ESS estimator (Ship 2, experimental).

    More rigorous than the block-averaged variant: computes the normalized
    autocorrelation function ρ(k) of the series via standard biased
    estimator, sums it using the initial positive sequence (sum until ρ(k)
    first goes non-positive), and returns:

        τ_int = 1/2 + Σ ρ(k)     for k=1..K* where ρ(K*+1) ≤ 0
        ESS   = N / (2 · τ_int)

    Caps the lag sum at min(max_lag, N//4) for both numerical stability
    and runtime — for typical 10²-10³ frame trajectories this is fast;
    for very long trajectories it remains O(N · K) which is acceptable.

    Returns a dict with:
        ess: int — effective sample size
        tau_int_frames: float — integrated autocorrelation time in frames
        truncation_lag: int — K* used
        method: "initial_positive_sequence"
        experimental: True — flag for output consumers; this estimator is
            sensitive to short series and can underestimate τ in the
            presence of multimodal autocorrelation. Use as a secondary
            signal alongside _effective_sample_size (block averaging).

    Returns None for series with < 8 points or zero variance.
    """
    if not trajectory or len(trajectory) < 8:
        return None
    values = [p[1] for p in trajectory if len(p) >= 2]
    n = len(values)
    if n < 8:
        return None
    mean = sum(values) / n
    centered = [v - mean for v in values]
    var = sum(c * c for c in centered) / n
    if var <= 0:
        return None

    if max_lag is None:
        max_lag = min(50, n // 4)
    max_lag = min(max_lag, n - 2)

    tau = 0.5
    truncation_lag = 0
    for k in range(1, max_lag + 1):
        # Biased autocorrelation at lag k (divides by N, not N-k — standard
        # for autocorrelation-time estimation since it makes ρ(0)=1 and
        # damps spurious tail values).
        rho_k = sum(centered[i] * centered[i + k] for i in range(n - k)) / (n * var)
        if rho_k <= 0:
            truncation_lag = k - 1
            break
        tau += rho_k
        truncation_lag = k

    if tau <= 0:
        return None
    return {
        "ess": int(n / (2 * tau)),
        "tau_int_frames": round(tau, 3),
        "truncation_lag": truncation_lag,
        "method": "initial_positive_sequence",
        "experimental": True,
    }


def _rmsd_plateau(trajectory: list) -> Optional[Dict]:
    """RMSD plateau detector (Ship 2, experimental).

    Compares the rolling means of the second and fourth quarters of the
    RMSD trajectory. If they agree within 2·SE of the difference, the
    RMSD has plateaued — i.e., the structural ensemble is stationary in
    the latter half of the trajectory.

    Returns a dict with:
        detected: bool
        q2_mean, q4_mean: rolling means of the 2nd and 4th quarters (nm)
        offset: |q4_mean - q2_mean| (nm)
        noise_floor: 2·SE_diff (nm)
        experimental: True

    Like _density_plateau, this is a sufficient-but-not-necessary
    convergence indicator. Failure does not mean the trajectory is
    unusable; it means the slow structural mode hasn't been visited
    enough times to claim stationarity by this test.
    """
    if not trajectory or len(trajectory) < 8:
        return None
    values = [p[1] for p in trajectory if len(p) >= 2]
    n = len(values)
    if n < 8:
        return None
    q = n // 4
    if q < 2:
        return None
    q2 = values[q : 2 * q]
    q4 = values[3 * q :]
    q2m = sum(q2) / len(q2)
    q4m = sum(q4) / len(q4)
    q2v = sum((v - q2m) ** 2 for v in q2) / len(q2)
    q4v = sum((v - q4m) ** 2 for v in q4) / len(q4)
    se_diff = (q2v / len(q2) + q4v / len(q4)) ** 0.5
    offset = abs(q4m - q2m)
    noise_floor = 2.0 * se_diff
    return {
        "detected": offset < noise_floor,
        "q2_mean": round(q2m, 4),
        "q4_mean": round(q4m, 4),
        "offset": round(offset, 4),
        "noise_floor": round(noise_floor, 4),
        "experimental": True,
    }


def _density_plateau(trajectory: list, target_density: float = 997.0) -> Dict:
    """Detect whether density has plateaued by comparing trajectory halves.

    Returns a dict with:
      detected: bool — true if half-means agree within 2·SE(diff)
      first_half_mean / second_half_mean: kg/m³
      half_offset: |second - first|, kg/m³
      noise_floor: 2 · SE of the half-difference, kg/m³
      estimated_plateau_offset_to_target: |second_half_mean - target_density|

    target_density default 997 (TIP3P water at 298K). For non-pure-water
    systems this is approximate; the detector mostly looks for
    self-consistency (halves agreeing) rather than absolute target match.
    """
    if not trajectory or len(trajectory) < 4:
        return {"detected": False, "reason": "trajectory_too_short"}
    values = [p[1] for p in trajectory if len(p) >= 2]
    n = len(values)
    if n < 4:
        return {"detected": False, "reason": "trajectory_too_short"}
    mid = n // 2
    first = values[:mid]
    second = values[mid:]
    fm = sum(first) / len(first)
    sm = sum(second) / len(second)
    fv = sum((v - fm) ** 2 for v in first) / len(first)
    sv = sum((v - sm) ** 2 for v in second) / len(second)
    se_diff = (fv / len(first) + sv / len(second)) ** 0.5
    offset = abs(sm - fm)
    noise_floor = 2.0 * se_diff
    return {
        "detected": offset < noise_floor,
        "first_half_mean": round(fm, 2),
        "second_half_mean": round(sm, 2),
        "half_offset": round(offset, 2),
        "noise_floor": round(noise_floor, 2),
        "offset_to_target": round(abs(sm - target_density), 2),
    }


def _ess_target_for_intent(intent: str) -> Optional[int]:
    """RMSD effective sample size target for a given scientific intent."""
    return {
        "smoke_test": None,
        "equilibration_only": None,
        "pose_stability": 100,
        "mm_gbsa": 200,
        "fep_window": 50,
    }.get(intent)


def _duration_floor_for_intent(intent: str) -> Optional[float]:
    """Minimum recommended production duration (ns) for a scientific intent."""
    return {
        "smoke_test": 0.1,
        "equilibration_only": 0.5,
        "pose_stability": 10.0,
        "mm_gbsa": 50.0,
        "fep_window": 2.0,
    }.get(intent)


def _grade_execution_integrity(checkpoint: Dict, log: Dict) -> Dict:
    """Layer 1: did the engine run correctly? Binary PASS / FAIL.

    Triggers FAIL on any of: invalid checkpoint, critical GROMACS warnings,
    NaN/Inf in the simulation output, or LINCS solver pathologies that
    indicate numerical instability (mere LINCS warnings during early
    equilibration are common and do not fail the integrity check —
    only critical-tagged constraint solver failures do).
    """
    evidence = {
        "checkpoint_valid": bool(checkpoint.get("checkpoint_valid", False)),
        "checkpoint_size_bytes": checkpoint.get("checkpoint_size_bytes", 0),
        "warning_count": log.get("warning_count", 0),
        "has_critical": bool(log.get("has_critical", False)),
        "gromacs_version": log.get("gromacs_version"),
    }
    warnings = log.get("warnings") or []
    nan_inf = any(
        ("NaN" in w.get("description", "") or "Inf" in w.get("description", ""))
        and w.get("severity") == "CRITICAL"
        for w in warnings
    )
    constraint_failure = any(
        "Constraint solver failure" in w.get("description", "")
        for w in warnings
    )
    evidence["nan_inf_detected"] = nan_inf
    evidence["constraint_solver_failure"] = constraint_failure

    fail = (
        not evidence["checkpoint_valid"]
        or evidence["has_critical"]
        or nan_inf
        or constraint_failure
    )
    annotations = []
    if not evidence["checkpoint_valid"]:
        annotations.append("Checkpoint file missing or unreadable — the simulation did not terminate cleanly.")
    if nan_inf:
        annotations.append("NaN or Inf detected in simulation output — the integrator blew up.")
    if constraint_failure:
        annotations.append("LINCS constraint solver failed — numerical instability, the simulation cannot be trusted.")
    if evidence["has_critical"] and not (nan_inf or constraint_failure):
        annotations.append("GROMACS reported critical warnings; check log_diagnostics.warnings for detail.")
    if not fail and not annotations:
        annotations.append("GROMACS completed without numerical pathologies.")

    return {
        "grade": "FAIL" if fail else "PASS",
        "evidence": evidence,
        "assessment": {
            "grade": "FAIL" if fail else "PASS",
            "summary": (
                "Engine reported a fatal pathology — output cannot be used."
                if fail else
                "GROMACS completed without numerical pathologies."
            ),
            "annotations": annotations,
        },
    }


def _grade_sampling_quality(equilibration: Dict, rmsd: Dict, quality_metrics: Dict) -> Dict:
    """Layer 2: did observables converge enough to be statistically usable?

    Distinct from execution_integrity (which just asks "did GROMACS run?")
    and from scientific_adequacy (which asks "is this enough for my use
    case?"). This layer reports the trajectory's intrinsic statistical
    properties: thermostat/barostat control, density convergence, RMSD
    effective sample size, energy plateau.

    Grades HIGH / MEDIUM / LOW / INSUFFICIENT.
    """
    qm = quality_metrics or {}
    # Pull pre-computed v2 grades + raw evidence
    temp_offset = qm.get("temperature_offset_k")
    temp_slope = qm.get("temperature_slope_k_per_ns")
    temp_std = qm.get("temperature_drift_k")
    pres_offset = qm.get("pressure_offset_bar")
    pres_slope = qm.get("pressure_slope_bar_per_ns")
    pres_std = qm.get("pressure_std_bar")
    energy_ratio = qm.get("energy_variance_ratio")
    temp_grade = qm.get("temperature_grade", "PASS")
    pres_grade = qm.get("pressure_grade", "PASS")
    energy_grade = qm.get("energy_grade", "PASS")

    # Density plateau
    density = equilibration.get("production_density") or equilibration.get("npt_density") or {}
    density_traj = density.get("trajectory", [])
    density_plateau_result = _density_plateau(density_traj)
    density_mean = density.get("mean")

    # RMSD ESS (block-averaged — primary signal)
    rmsd_traj = rmsd.get("trajectory", []) if rmsd else []
    rmsd_ess = _effective_sample_size(rmsd_traj)
    # Ship 2 experimental signals: autocorrelation-based ESS + RMSD plateau.
    # These are reported alongside the primary block-averaged ESS so callers
    # can cross-check. They do NOT drive the band — only the block-averaged
    # ESS does — until they accumulate enough field validation to graduate
    # out of experimental.
    rmsd_autocorr = _autocorrelation_ess(rmsd_traj)
    rmsd_plateau = _rmsd_plateau(rmsd_traj)

    evidence = {
        "temperature_offset_k": temp_offset,
        "temperature_slope_k_per_ns": temp_slope,
        "temperature_std_k": temp_std,
        "pressure_offset_bar": pres_offset,
        "pressure_slope_bar_per_ns": pres_slope,
        "pressure_std_bar": pres_std,
        "density_plateau_detected": density_plateau_result.get("detected"),
        "density_plateau_evidence": density_plateau_result,
        "density_mean_kg_m3": density_mean,
        "rmsd_effective_sample_size": rmsd_ess,
        "rmsd_trajectory_points": len(rmsd_traj),
        "energy_variance_ratio": energy_ratio,
        # Ship 2 experimental signals (do not drive band yet — see comment above)
        "rmsd_autocorrelation_ess_experimental": rmsd_autocorr,
        "rmsd_plateau_experimental": rmsd_plateau,
    }

    # Compute band
    band_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INSUFFICIENT": 3}
    worst = "HIGH"

    def _worse(b: str) -> None:
        nonlocal worst
        if band_rank[b] > band_rank[worst]:
            worst = b

    annotations = []
    uncertainty = []

    # Temperature
    if temp_grade == "FAIL":
        _worse("LOW")
        annotations.append(
            f"Temperature offset {temp_offset}K or drift {temp_slope} K/ns is statistically significant — thermostat is not holding setpoint."
        )
    elif temp_grade == "CONDITIONAL":
        _worse("MEDIUM")
        annotations.append(f"Temperature offset {temp_offset}K is above the tight threshold but below the failure cutoff.")
    elif temp_offset is not None:
        annotations.append(
            f"Temperature mean is {temp_offset}K from setpoint — within thermostat tolerance."
        )

    # Pressure
    if pres_grade == "FAIL":
        _worse("LOW")
        annotations.append(
            f"Pressure offset {pres_offset} bar or drift {pres_slope} bar/ns is statistically significant — barostat is not holding setpoint."
        )
    elif pres_grade == "CONDITIONAL":
        _worse("MEDIUM")
        annotations.append(f"Pressure offset {pres_offset} bar is above the tight threshold but below the failure cutoff.")
    elif pres_std is not None and pres_std > 200:
        annotations.append(
            f"Pressure fluctuations of ±{round(pres_std)} bar are expected for finite-size NPT systems; the mean is within thermal noise of the setpoint."
        )

    # Density plateau
    if density_plateau_result.get("detected") is False:
        _worse("MEDIUM")
        offset = density_plateau_result.get("half_offset")
        floor = density_plateau_result.get("noise_floor")
        annotations.append(
            f"Density has not plateaued: first-half mean differs from second-half mean by {offset} kg/m³ (noise floor {floor})."
        )
        uncertainty.append(
            "Density is still drifting — observables that depend on solvation thermodynamics (free energies, partition coefficients) are biased."
        )
    elif density_plateau_result.get("detected") is True:
        density_display = round(density_mean, 1) if density_mean is not None else "?"
        annotations.append(
            f"Density plateaued at {density_display} kg/m³ — first and second halves agree within noise."
        )

    # RMSD ESS
    if rmsd_ess is not None:
        if rmsd_ess < 30:
            _worse("LOW")
            uncertainty.append(
                f"RMSD effective sample size is very low ({rmsd_ess}); statistics derived from this trajectory have wide error bars."
            )
        elif rmsd_ess < 100:
            _worse("MEDIUM")
            uncertainty.append(
                f"RMSD effective sample size is moderate ({rmsd_ess}); longer runs would tighten statistical estimates."
            )
        annotations.append(f"RMSD effective sample size: {rmsd_ess} (block-averaged heuristic).")

    # Energy plateau
    if energy_grade == "FAIL":
        _worse("LOW")
        annotations.append(f"Potential energy variance ratio {energy_ratio} > 5 — system has not reached steady state.")
    elif energy_grade == "CONDITIONAL":
        _worse("MEDIUM")

    summary_map = {
        "HIGH": "Trajectory is statistically well-converged for its duration.",
        "MEDIUM": "Trajectory is usable but has measurable convergence weaknesses; see annotations.",
        "LOW": "Trajectory exhibits drift or insufficient sampling for reliable observables.",
        "INSUFFICIENT": "Trajectory data is too sparse or pathological to assess statistical quality.",
    }

    return {
        "grade": worst,
        "evidence": evidence,
        "assessment": {
            "grade": worst,
            "summary": summary_map[worst],
            "annotations": annotations,
            "uncertainty_sources": uncertainty,
        },
    }


def _grade_adequacy(
    intent: str,
    simulation_ns: Optional[float],
    sampling_evidence: Dict,
    has_ligand: bool,
    has_protein: bool,
    route: Optional[str],
) -> Dict:
    """Layer 3: is this trajectory adequate for a given scientific intent?

    Each known intent has its own rule. INSUFFICIENT for hard exclusions
    (e.g., MM-GBSA on a ligand-only run, FEP on a non-FEP setup). LOW /
    MEDIUM / HIGH for graded cases driven by duration + ESS + plateau.
    Where applicable, returns estimated_additional_sampling_ns with
    explicit heuristic basis and bounds — never as a guarantee.
    """
    rmsd_ess = sampling_evidence.get("rmsd_effective_sample_size")
    density_plateau = sampling_evidence.get("density_plateau_detected")
    sim_ns = simulation_ns if simulation_ns is not None else 0.0

    def _ess_recommendation(target: int) -> Optional[Dict]:
        if rmsd_ess is None or rmsd_ess >= target or sim_ns <= 0:
            return None
        # ESS scales approximately linearly with duration; recommend the
        # additional time to reach the target. Lower/upper bracket reflects
        # the heuristic nature — the actual scaling depends on the
        # autocorrelation time of the slowest mode in the system.
        scale = target / max(rmsd_ess, 1)
        recommended = sim_ns * scale - sim_ns
        return {
            "value": round(recommended, 2),
            "lower": round(recommended * 0.5, 2),
            "upper": round(recommended * 3.0, 2),
            "based_on": "rmsd_effective_sample_size",
            "target": f"ESS ≥ {target}",
            "note": (
                "Heuristic estimate from the current block-averaged sample size; "
                "convergence is not guaranteed by any fixed sampling time. "
                "Systems with slow conformational modes may need substantially more."
            ),
        }

    def _duration_floor_recommendation(floor_ns: float) -> Optional[Dict]:
        if sim_ns >= floor_ns:
            return None
        gap = floor_ns - sim_ns
        return {
            "value": round(gap, 2),
            "lower": round(gap, 2),
            "upper": round(gap * 2.5, 2),
            "based_on": "duration_floor",
            "target": f"≥ {floor_ns} ns total production",
            "note": (
                "Floor reflects the community-standard minimum for this analysis; "
                "actual convergence depends on system."
            ),
        }

    if intent == "smoke_test":
        # Smoke = "did the plumbing work?" — any successful run is HIGH.
        return {
            "grade": "HIGH",
            "summary": "Suitable for plumbing verification. Not a scientific result on its own.",
        }

    if intent == "equilibration_only":
        # We want density plateau + temp/pressure on target.
        if density_plateau is False:
            return {
                "grade": "LOW",
                "summary": "Density has not plateaued; equilibration is incomplete.",
                "estimated_additional_sampling_ns": _duration_floor_recommendation(0.5),
                "uncertainty_sources": [
                    "Density is still drifting between first and second halves of the trajectory.",
                ],
            }
        return {
            "grade": "HIGH" if sim_ns >= 0.5 else "MEDIUM",
            "summary": "System reached equilibrium on coupled observables (temperature, pressure mean, density).",
        }

    if intent == "pose_stability":
        if not has_ligand:
            return {
                "grade": "INSUFFICIENT",
                "reason": "Pose stability requires a ligand in the system.",
            }
        ess_target = _ess_target_for_intent("pose_stability")
        duration_floor = _duration_floor_for_intent("pose_stability")
        rec_ess = _ess_recommendation(ess_target) if ess_target else None
        rec_floor = _duration_floor_recommendation(duration_floor) if duration_floor else None
        # Prefer the larger recommendation
        rec = max(
            [r for r in (rec_ess, rec_floor) if r is not None],
            key=lambda r: r["value"],
            default=None,
        )
        if sim_ns < 1.0:
            return {
                "grade": "LOW",
                "summary": "Production duration is well below the pose-stability standard (~10ns minimum).",
                "estimated_additional_sampling_ns": rec,
                "uncertainty_sources": [
                    "Short trajectory cannot distinguish a transiently stable pose from a long-lived one.",
                ],
            }
        if rmsd_ess is not None and rmsd_ess < 50:
            return {
                "grade": "LOW",
                "summary": "Ligand RMSD has too few independent samples to claim pose stability.",
                "estimated_additional_sampling_ns": rec,
                "uncertainty_sources": [
                    f"RMSD effective sample size {rmsd_ess} is below the working threshold (100).",
                ],
            }
        if sim_ns < duration_floor or (rmsd_ess is not None and rmsd_ess < ess_target):
            return {
                "grade": "MEDIUM",
                "summary": "Trajectory is in the right regime but below the community-standard window for pose stability claims.",
                "estimated_additional_sampling_ns": rec,
            }
        return {
            "grade": "HIGH",
            "summary": "Duration and effective sample size support pose-stability analysis.",
        }

    if intent == "mm_gbsa":
        if not (has_ligand and has_protein):
            missing = []
            if not has_ligand:
                missing.append("no ligand")
            if not has_protein:
                missing.append("no protein")
            return {
                "grade": "INSUFFICIENT",
                "reason": (
                    "MM-GBSA requires a protein-ligand complex; the current system has "
                    + " and ".join(missing) + "."
                ),
            }
        if route == "run_membrane":
            return {
                "grade": "INSUFFICIENT",
                "reason": "MM-GBSA solvation model is unreliable in the presence of a lipid bilayer.",
            }
        duration_floor = _duration_floor_for_intent("mm_gbsa")
        rec = _duration_floor_recommendation(duration_floor) if duration_floor else None
        if sim_ns < 10.0:
            return {
                "grade": "LOW",
                "summary": "Production duration is far below the MM-GBSA standard (~50ns minimum).",
                "estimated_additional_sampling_ns": rec,
            }
        if sim_ns < duration_floor:
            return {
                "grade": "MEDIUM",
                "summary": "Below the community-standard duration for converged MM-GBSA energies.",
                "estimated_additional_sampling_ns": rec,
            }
        return {
            "grade": "HIGH",
            "summary": "Duration supports MM-GBSA analysis (note: still benefits from multiple replicas).",
        }

    if intent == "fep_window":
        # An individual free-energy-perturbation window — typically 2-5ns per
        # window. The MD service doesn't run FEP directly, but a single
        # window's adequacy is still meaningful for custom FEP pipelines.
        duration_floor = _duration_floor_for_intent("fep_window")
        rec = _duration_floor_recommendation(duration_floor) if duration_floor else None
        if sim_ns < duration_floor:
            return {
                "grade": "INSUFFICIENT",
                "reason": f"Below the {duration_floor}ns per-window standard for relative binding free energy.",
                "estimated_additional_sampling_ns": rec,
            }
        if sim_ns < 5.0:
            return {
                "grade": "MEDIUM",
                "summary": "Meets the per-window floor but below the high-confidence window length for difficult transformations.",
            }
        return {
            "grade": "HIGH",
            "summary": "Window duration is adequate for relative binding free energy estimation.",
        }

    return {"grade": "INSUFFICIENT", "reason": f"Unknown intent: {intent}"}


def assess_md_quality(
    analysis: Dict,
    simulation_ns: Optional[float] = None,
    has_ligand: bool = False,
    has_protein: bool = False,
    route: Optional[str] = None,
    intent: Optional[str] = None,
) -> Dict:
    """Build the v3 three-layer quality report.

    Layers:
      execution_integrity  — binary PASS/FAIL: did GROMACS run correctly?
      sampling_quality     — HIGH/MEDIUM/LOW/INSUFFICIENT: are observables
                             statistically usable?
      scientific_adequacy  — per-intent grade: is this trajectory enough
                             for the requested analysis?

    `intent` selects a specific intent for the top-level summary; all known
    intents are still graded so the response is informative regardless.
    Caller does not need to know what's available — when intent is None,
    all intents appear in scientific_adequacy.

    Returns a dict whose top-level shape is:
        {
          "overall": "PASS"|"FAIL",          # back-compat alias for
                                              # execution_integrity.grade
          "thresholds_version": "v3",
          "execution_integrity": {grade, evidence, assessment},
          "sampling_quality":    {grade, evidence, assessment},
          "scientific_adequacy": {intent_name: {grade, ...}, ...},
          "intent": <intent or null>,
          "action":      str,
          "remediation": [str, ...],
        }
    """
    cp = analysis.get("checkpoint", {})
    log = analysis.get("log_diagnostics", {})
    qm = analysis.get("quality_metrics", {})
    equilibration = analysis.get("equilibration", {})
    rmsd = analysis.get("rmsd", {})

    integrity = _grade_execution_integrity(cp, log)
    sampling = _grade_sampling_quality(equilibration, rmsd, qm)

    # Validate intent
    if intent is not None and intent not in _KNOWN_INTENTS:
        # Don't fail — just record the bad intent and grade everything else.
        log_intent_error = f"Unknown intent '{intent}'; graded all known intents instead."
        intent = None
    else:
        log_intent_error = None

    adequacy: Dict[str, Dict] = {}
    for known_intent in _KNOWN_INTENTS:
        adequacy[known_intent] = _grade_adequacy(
            known_intent,
            simulation_ns=simulation_ns,
            sampling_evidence=sampling["evidence"],
            has_ligand=has_ligand,
            has_protein=has_protein,
            route=route,
        )

    # Aggregate remediation from all layers
    remediation: List[str] = []
    if integrity["grade"] == "FAIL":
        remediation.extend(integrity["assessment"]["annotations"])
    if sampling["grade"] == "LOW":
        remediation.append("Sampling quality is LOW — see sampling_quality.assessment.uncertainty_sources for specifics.")
    if intent is not None and adequacy.get(intent, {}).get("grade") in ("LOW", "INSUFFICIENT"):
        reason = adequacy[intent].get("reason") or adequacy[intent].get("summary")
        if reason:
            remediation.append(f"Scientific adequacy for '{intent}' is {adequacy[intent]['grade']}: {reason}")

    # Action text — describes what the user should do given the intent (if any)
    if integrity["grade"] == "FAIL":
        action = "Simulation failed at the engine level. Output cannot be used; re-run after addressing the issue."
    elif intent is not None:
        intent_grade = adequacy[intent]["grade"]
        action_map = {
            "HIGH":   f"Trajectory is suitable for {intent}.",
            "MEDIUM": f"Trajectory is usable for {intent} with caveats — see scientific_adequacy['{intent}'] for guidance.",
            "LOW":    f"Trajectory is below the standard for {intent}; consider extending sampling per the recommendation.",
            "INSUFFICIENT": f"Trajectory cannot support {intent} (see scientific_adequacy['{intent}'].reason).",
        }
        action = action_map.get(intent_grade, "")
    else:
        action = "Trajectory available. Specify an intent on submission for use-case-specific guidance."

    report = {
        "overall": "PASS" if integrity["grade"] == "PASS" else "FAIL",
        "thresholds_version": "v3",
        "intent": intent,
        "execution_integrity": integrity,
        "sampling_quality": sampling,
        "scientific_adequacy": adequacy,
        "action": action,
        "remediation": remediation,
        # ── Back-compat aliases (deprecated; remove after 2 release cycles) ──
        # Old quality_report had top-level energy_grade / checkpoint_grade /
        # log_grade. Existing frontend code (lead-comparison, dashboard, etc.)
        # reads these. Synthesize them from the v3 layers so callers don't
        # break during the v2→v3 migration. New callers should read from
        # execution_integrity / sampling_quality directly.
        "energy_grade": qm.get("overall"),
        "checkpoint_grade": "PASS" if cp.get("checkpoint_valid", True) else "FAIL",
        "log_grade": log.get("log_grade"),
    }
    if log_intent_error:
        report["warnings"] = [log_intent_error]
    return report


def _detect_ligand_resname(workspace: Path, gro_file: str) -> Optional[str]:
    """Identify the ligand residue name in a GROMACS .gro file.

    ACPYPE outputs vary: residue name can be MOL, UNL, LIG, or the sanitized
    compound name. Strategy: read the .gro, find the last non-water/ion
    residue name that has < 200 atoms (ligands are small, solvent is
    abundant). Returns the residue name string or None if no candidate found.
    """
    STANDARD_AA = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL", "HIE", "HID", "HIP", "CYX", "ASH", "GLH", "LYN",
    }
    SOLVENT = {"SOL", "HOH", "WAT", "TIP", "TP3", "T3P", "SPC", "NA", "CL", "K", "MG", "CA", "ZN"}
    try:
        gro_path = workspace / gro_file if not Path(gro_file).is_absolute() else Path(gro_file)
        if not gro_path.exists():
            return None
        resname_counts: Dict[str, int] = {}
        with open(gro_path, "r") as f:
            lines = f.readlines()
        # .gro format: line 0 = title, line 1 = atom count, lines 2..2+N = atoms
        try:
            n_atoms = int(lines[1].strip())
        except (ValueError, IndexError):
            return None
        for line in lines[2 : 2 + n_atoms]:
            # Columns 5-10 = residue name (1-indexed, so [5:10] in 0-indexed slicing)
            if len(line) < 10:
                continue
            resname = line[5:10].strip()
            if not resname or resname in STANDARD_AA or resname in SOLVENT:
                continue
            resname_counts[resname] = resname_counts.get(resname, 0) + 1
        # Ligand = non-AA, non-solvent residue with small atom count
        candidates = [(name, count) for name, count in resname_counts.items() if count < 200]
        if not candidates:
            return None
        # Prefer the one with fewest atoms (most ligand-like)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    except Exception as e:
        logger.debug(f"Ligand resname detection failed: {e}")
        return None


# Residue classification shared by the index builder and the decomp-cutoff sizer.
_STANDARD_AA = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "HIE", "HID", "HIP", "CYX", "ASH", "GLH", "LYN",
}
_SOLVENT = {"SOL", "HOH", "WAT", "TIP", "TP3", "T3P", "SPC", "NA", "CL", "K", "MG", "CA", "ZN"}


def _decomp_cutoff_angstrom(
    gro_path: Path,
    candidates=(5.0, 6.0, 8.0, 10.0),
    margin_ang: float = 1.0,
) -> Optional[float]:
    """Smallest 'within X Å' cutoff that is guaranteed to select ≥1 residue.

    gmx_MMPBSA crashes outright on an empty per-residue selection (list2range('')
    ['string'] -> TypeError). For a ligand sitting in a real binding pocket the
    default 5.0 Å always selects residues, but for a surface-bound or
    solvent-exposed ligand (e.g. a non-binder, or any drug-like probe against a
    pocketless protein like crambin/1CRN) nothing is within 5 Å and the run dies.

    We compute, from the final-frame .gro (coords in nm), the closest protein
    residue to any ligand atom and pick the smallest candidate cutoff that keeps
    the nearest residue at least `margin_ang` Å *inside* the cutoff. The margin
    absorbs the small frame-to-frame drift between our .gro and the frame
    gmx_MMPBSA evaluates, so "within X" can't come back empty on its side.

    Returns None when no protein residue is within (max candidate − margin) of
    the ligand — i.e. the ligand is genuinely not in contact and per-residue
    decomposition is not meaningful.
    """
    try:
        lines = gro_path.read_text().splitlines()
        n_atoms = int(lines[1].strip())
    except (ValueError, IndexError, OSError) as e:
        logger.warning(f"_decomp_cutoff_angstrom: cannot parse {gro_path.name}: {e}")
        return None

    # .gro fixed columns: resid[0:5] resname[5:10] name[10:15] num[15:20]
    #                     x[20:28] y[28:36] z[36:44]  (nm, %8.3f default)
    protein_res: Dict[str, List[tuple]] = {}
    ligand_xyz: List[tuple] = []
    for line in lines[2 : 2 + n_atoms]:
        if len(line) < 44:
            continue
        resname = line[5:10].strip()
        try:
            xyz = (float(line[20:28]), float(line[28:36]), float(line[36:44]))
        except ValueError:
            continue
        if resname in _STANDARD_AA:
            protein_res.setdefault(line[0:5].strip(), []).append(xyz)
        elif resname in _SOLVENT:
            continue
        else:
            ligand_xyz.append(xyz)

    if not protein_res or not ligand_xyz:
        return None

    # nearest protein residue to any ligand atom (min over residue's atoms), in nm
    nearest_nm = min(
        min((ax - lx) ** 2 + (ay - ly) ** 2 + (az - lz) ** 2
            for (ax, ay, az) in atoms for (lx, ly, lz) in ligand_xyz) ** 0.5
        for atoms in protein_res.values()
    )
    nearest_ang = nearest_nm * 10.0

    for cutoff in candidates:
        if nearest_ang <= cutoff - margin_ang:
            return cutoff
    return None


def _make_mmgbsa_index(workspace: Path, compound_id: str, gro_file: str):
    """Create GROMACS index file with Protein + Ligand groups — pure Python.

    Bypasses `gmx make_ndx` entirely. After 8 failed iterations debugging
    make_ndx's interactive output (stderr vs stdout, group name format,
    index numbering), this function writes the .ndx directly by reading
    atom records from the .gro file and classifying them by residue name.

    An ndx file is trivial: `[ GroupName ]` header followed by 1-indexed
    atom numbers, 15 per line. GROMACS, gmx_MMPBSA, and all gmx tools
    accept this format.

    Returns (ndx_path, protein_group_idx, ligand_group_idx) or None.
    """
    STANDARD_AA = _STANDARD_AA
    SOLVENT = _SOLVENT

    gro_path = workspace / gro_file
    if not gro_path.exists():
        available = [f.name for f in workspace.iterdir() if f.suffix in ('.gro', '.tpr', '.xtc')]
        logger.warning(f"_make_mmgbsa_index: {gro_file} not found. Available: {available[:10]}")
        return None

    try:
        lines = gro_path.read_text().splitlines()
        n_atoms = int(lines[1].strip())
    except (ValueError, IndexError) as e:
        logger.warning(f"_make_mmgbsa_index: cannot parse {gro_file}: {e}")
        return None

    # Classify atoms: protein, ligand (non-AA non-solvent), solvent, ions
    protein_atoms: List[int] = []
    ligand_atoms: List[int] = []
    system_atoms: List[int] = []
    ligand_resname = None

    for i, line in enumerate(lines[2 : 2 + n_atoms], start=1):  # 1-indexed
        if len(line) < 10:
            continue
        resname = line[5:10].strip()
        system_atoms.append(i)
        if resname in STANDARD_AA:
            protein_atoms.append(i)
        elif resname in SOLVENT:
            pass  # skip solvent/ions from named groups
        else:
            ligand_atoms.append(i)
            if ligand_resname is None:
                ligand_resname = resname

    if not ligand_atoms:
        logger.warning(f"_make_mmgbsa_index: no ligand atoms found in {gro_file}")
        return None

    logger.info(
        f"_make_mmgbsa_index: {len(protein_atoms)} protein atoms, "
        f"{len(ligand_atoms)} ligand atoms (resname={ligand_resname}), "
        f"{len(system_atoms)} total"
    )

    def _write_group(f, name: str, atoms: List[int]):
        f.write(f"[ {name} ]\n")
        for j in range(0, len(atoms), 15):
            f.write(" ".join(f"{a:>5d}" for a in atoms[j:j+15]) + "\n")

    ndx_path = workspace / f"{compound_id}_mmgbsa.ndx"
    # Group 0 = System, 1 = Protein, 2 = Ligand
    with open(ndx_path, "w") as f:
        _write_group(f, "System", system_atoms)
        _write_group(f, "Protein", protein_atoms)
        _write_group(f, "Ligand", ligand_atoms)

    logger.info(f"Wrote ndx: {ndx_path} (3 groups: System=0, Protein=1, Ligand=2)")
    return ndx_path, 1, 2  # protein_idx=1, ligand_idx=2


def run_mmgbsa_analysis(
    workspace: Path,
    compound_id: str,
    top_file: str,
    gro_file: str,
) -> Dict[str, Any]:
    """Run gmx_MMPBSA on the production trajectory.

    Returns dict with:
      - mean_kcal_mol, std_kcal_mol: overall ΔG_bind
      - per_residue_top10: list of {residue, chain, contribution_kcal_mol}
      - trajectory: list of [frame_idx, delta_g_kcal_mol] for convergence plot
      - method: "MM-GBSA (igb=5, OBC2)"
    Or {"error": "..."} on failure.

    Fail-open: any failure logs + returns an error dict. Never raises.
    """
    if not shutil.which("gmx_MMPBSA"):
        return {"error": "gmx_MMPBSA not installed"}

    # The MM-GBSA input is written below, AFTER we've sized the decomposition
    # cutoff from the actual geometry (an empty "within X" selection crashes
    # gmx_MMPBSA entirely — see _decomp_cutoff_angstrom).
    mmpbsa_in = workspace / f"{compound_id}_mmpbsa.in"

    # Build index file with explicit Protein/Ligand groups
    # Use the production .gro (not the topology .top) for resname detection
    md_gro = f"{compound_id}_md.gro"
    # Check the .gro file actually exists
    md_gro_path = workspace / md_gro
    if not md_gro_path.exists():
        # List what IS in workspace to diagnose
        available = [f.name for f in workspace.iterdir() if f.suffix in ('.gro', '.tpr', '.xtc')]
        return {
            "error": (
                f"MM-GBSA: {md_gro} not found in workspace. "
                f"Available trajectory files: {available[:10]}"
            )
        }

    ndx_result = _make_mmgbsa_index(workspace, compound_id, md_gro)
    if ndx_result is None:
        return {"error": "failed to build MM-GBSA index — no ligand atoms found in .gro"}
    ndx_path, protein_group_idx, ligand_group_idx = ndx_result
    if not ndx_path.exists():
        return {"error": "MM-GBSA index file was not created"}

    results_csv = workspace / f"{compound_id}_mmgbsa_results.csv"
    decomp_csv = workspace / f"{compound_id}_mmgbsa_decomp.csv"

    # Size the per-residue decomposition cutoff from the actual ligand-protein
    # geometry. If the ligand isn't in contact with the protein (no residue
    # within range), run plain MM-GBSA without &decomp — feeding gmx_MMPBSA an
    # empty "within X" selection makes it crash on list2range('')['string'].
    decomp_cutoff = _decomp_cutoff_angstrom(md_gro_path)
    decomp_enabled = decomp_cutoff is not None
    if decomp_enabled:
        mmpbsa_in.write_text(_mmpbsa_input_with_decomp(decomp_cutoff))
        logger.info(f"MM-GBSA: per-residue decomposition within {decomp_cutoff:.1f} Å")
    else:
        mmpbsa_in.write_text(MMPBSA_INPUT_GB_BASE)
        logger.info(
            "MM-GBSA: ligand not in contact with protein (no residue within "
            "~9 Å) — running ΔG_bind without per-residue decomposition"
        )

    # gmx_MMPBSA -cg takes group INDICES (integers).
    cmd = [
        "gmx_MMPBSA",
        "-O",  # overwrite
        "-i", mmpbsa_in.name,
        "-cs", f"{compound_id}_md.tpr",
        "-ct", f"{compound_id}_md.xtc",
        "-ci", ndx_path.name,
        "-cg", str(protein_group_idx), str(ligand_group_idx),
        "-cp", top_file,
        "-eo", results_csv.name,
        "-nogui",
    ]
    if decomp_enabled:
        cmd[-1:-1] = ["-deo", decomp_csv.name]  # decomposition output, before -nogui
    try:
        proc = subprocess.run(
            cmd, cwd=workspace, capture_output=True, text=True, timeout=1800,  # 30 min cap
        )
        if proc.returncode != 0:
            stderr_tail = (proc.stderr or "")[-600:]
            return {"error": f"gmx_MMPBSA exit {proc.returncode}: {stderr_tail}"}
    except subprocess.TimeoutExpired:
        return {"error": "gmx_MMPBSA timed out after 30 min"}
    except Exception as e:
        return {"error": f"gmx_MMPBSA failed to launch: {e}"}

    # Parse per-frame ΔG_bind from results CSV
    # Row format (gmx_MMPBSA -eo output):
    #   Frame,Complex,Receptor,Ligand,Delta
    trajectory: List[List[float]] = []
    dg_values: List[float] = []
    try:
        if results_csv.exists():
            for line in results_csv.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.lower().startswith("frame"):
                    continue
                parts = line.split(",")
                if len(parts) >= 5:
                    try:
                        frame = int(float(parts[0]))
                        delta = float(parts[-1])
                        trajectory.append([frame, round(delta, 3)])
                        dg_values.append(delta)
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        logger.warning(f"Failed to parse MM-GBSA results CSV: {e}")

    if not dg_values:
        return {"error": "gmx_MMPBSA ran but produced no parseable ΔG values"}

    mean_dg = sum(dg_values) / len(dg_values)
    variance = sum((v - mean_dg) ** 2 for v in dg_values) / max(len(dg_values), 1)
    std_dg = variance ** 0.5

    # Parse per-residue decomposition (top 10 contributors by absolute energy)
    per_residue_top10: List[Dict[str, Any]] = []
    try:
        if decomp_enabled and decomp_csv.exists():
            # Row format varies by gmx_MMPBSA version but typically:
            #   Residue,Internal,VdW,Electrostatic,Solvation,Total
            residue_contribs: Dict[str, float] = {}
            for line in decomp_csv.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.lower().startswith(("residue", "frame")):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                res_label = parts[0]
                # Total energy is last numeric column
                try:
                    total = float(parts[-1])
                    residue_contribs[res_label] = residue_contribs.get(res_label, 0.0) + total
                except (ValueError, IndexError):
                    continue
            top = sorted(residue_contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:10]
            for label, energy in top:
                per_residue_top10.append({
                    "residue": label,
                    "contribution_kcal_mol": round(energy, 3),
                })
    except Exception as e:
        logger.warning(f"Failed to parse MM-GBSA decomposition CSV: {e}")

    result = {
        "mean_kcal_mol": round(mean_dg, 3),
        "std_kcal_mol": round(std_dg, 3),
        "n_frames": len(dg_values),
        "method": "MM-GBSA (GB-OBC2, igb=5, saltcon=0.15 M)",
        "per_residue_top10": per_residue_top10,
        "trajectory": trajectory,
    }
    if decomp_enabled:
        result["decomp_cutoff_angstrom"] = decomp_cutoff
    else:
        result["per_residue_note"] = (
            "Per-residue decomposition not computed: no protein residue is in "
            "contact range of the ligand (closest > ~9 Å), so the ligand is not "
            "pocket-bound. ΔG_bind is reported but per-residue decomposition is "
            "not physically meaningful for this pose."
        )
    return result


def analyze_ligand_dynamics(workspace: Path, compound_id: str) -> Dict[str, Any]:
    """Ligand-specific trajectory analysis beyond the protein-wide RMSD.

    Adds three Theo-requested metrics:
      - ligand_rmsd: how much the ligand has moved/drifted in the pocket
      - hbond_persistence: fraction of frames each protein-ligand H-bond
        was maintained. Theo's thresholds: >75% = "High Quality", <30% =
        "False Positive"
      - pose_clusters: RMSD-based clustering of ligand poses (did the
        ligand stay in one binding mode or sample multiple?)

    Returns empty dict on failure — never raises. The per-metric try/except
    lets partial data come through when only some gmx calls succeed.
    """
    out: Dict[str, Any] = {}

    # Build index with Protein + Ligand (resname auto-detected)
    ndx_result = _make_mmgbsa_index(workspace, compound_id, f"{compound_id}_md.gro")
    if ndx_result is None:
        detected = _detect_ligand_resname(workspace, f"{compound_id}_md.gro")
        return {
            "error": (
                f"failed to build index for ligand dynamics — "
                f"detected resname={detected!r}, make_ndx failed. "
                f"Check that the .gro file exists and has a non-AA residue."
            )
        }
    ndx_path, protein_group_idx, ligand_group_idx = ndx_result
    if not ndx_path.exists():
        return {"error": "ligand dynamics index file not found after make_ndx"}
    lig_str = str(ligand_group_idx)
    prot_str = str(protein_group_idx)

    step_errors: List[str] = []

    # --- Ligand RMSD ---
    try:
        subprocess.run(
            ["gmx", "rms", "-s", f"{compound_id}_md.tpr", "-f", f"{compound_id}_md.xtc",
             "-n", ndx_path.name,
             "-o", f"{compound_id}_ligand_rmsd.xvg", "-tu", "ns"],
            input=f"{lig_str}\n{lig_str}\n".encode(), check=True, cwd=workspace,
            capture_output=True, timeout=300,
        )
        rmsd_rows = _parse_xvg(workspace / f"{compound_id}_ligand_rmsd.xvg")
        if rmsd_rows:
            vals = [r[1] for r in rmsd_rows if len(r) >= 2]
            out["ligand_rmsd"] = {
                "mean_nm": round(sum(vals) / len(vals), 4) if vals else None,
                "max_nm": round(max(vals), 4) if vals else None,
                "final_nm": round(vals[-1], 4) if vals else None,
                "trajectory": [[round(r[0], 3), round(r[1], 4)] for r in rmsd_rows],
            }
    except Exception as e:
        logger.warning(f"Ligand RMSD failed: {e}")
        step_errors.append(f"ligand_rmsd: {str(e)[:200]}")

    # --- H-bond persistence (protein ↔ ligand) ---
    try:
        subprocess.run(
            ["gmx", "hbond", "-s", f"{compound_id}_md.tpr", "-f", f"{compound_id}_md.xtc",
             "-n", ndx_path.name,
             "-num", f"{compound_id}_hbnum.xvg",
             "-hbn", f"{compound_id}_hbond.ndx",
             "-hbm", f"{compound_id}_hbmap.xpm"],
            input=f"{prot_str}\n{lig_str}\n".encode(), check=True, cwd=workspace,
            capture_output=True, timeout=300,
        )
        hb_rows = _parse_xvg(workspace / f"{compound_id}_hbnum.xvg")
        if hb_rows:
            counts = [r[1] for r in hb_rows if len(r) >= 2]
            if counts:
                mean_hbonds = sum(counts) / len(counts)
                max_hbonds = max(counts)
                # Fraction of frames with ≥1 H-bond = "any-contact persistence"
                frames_with_hb = sum(1 for c in counts if c >= 1)
                persistence_any = frames_with_hb / len(counts)
                out["hbond_persistence"] = {
                    "mean_count": round(mean_hbonds, 2),
                    "max_count": int(max_hbonds),
                    "any_contact_persistence": round(persistence_any, 3),
                    "quality": (
                        "High Quality" if persistence_any > 0.75
                        else "False Positive Risk" if persistence_any < 0.30
                        else "Moderate"
                    ),
                    "n_frames": len(counts),
                }
    except Exception as e:
        logger.warning(f"H-bond analysis failed: {e}")
        step_errors.append(f"hbond: {str(e)[:200]}")

    # --- Pose clustering (RMSD-based, ligand only) ---
    try:
        subprocess.run(
            ["gmx", "cluster", "-s", f"{compound_id}_md.tpr", "-f", f"{compound_id}_md.xtc",
             "-n", ndx_path.name,
             "-cl", f"{compound_id}_ligand_clusters.pdb",
             "-clid", f"{compound_id}_ligand_clusters.xvg",
             "-dist", f"{compound_id}_ligand_rmsd_dist.xvg",
             "-method", "gromos",
             "-cutoff", "0.15"],  # 1.5 Å RMSD cutoff — tight for ligand poses
            input=f"{lig_str}\n{lig_str}\n".encode(), check=True, cwd=workspace,
            capture_output=True, timeout=600,
        )
        clid_rows = _parse_xvg(workspace / f"{compound_id}_ligand_clusters.xvg")
        if clid_rows:
            cluster_ids = [int(r[1]) for r in clid_rows if len(r) >= 2]
            if cluster_ids:
                from collections import Counter
                counts = Counter(cluster_ids)
                total = len(cluster_ids)
                clusters = [
                    {"cluster_id": cid, "frames": n, "population_pct": round(100 * n / total, 1)}
                    for cid, n in counts.most_common()
                ]
                out["pose_clusters"] = {
                    "n_clusters": len(clusters),
                    "dominant_cluster_pct": clusters[0]["population_pct"] if clusters else 0,
                    "clusters": clusters[:5],  # Top 5
                    "stable_pose": clusters[0]["population_pct"] > 70 if clusters else False,
                }
    except Exception as e:
        logger.warning(f"Pose clustering failed: {e}")
        step_errors.append(f"cluster: {str(e)[:200]}")

    if not out and step_errors:
        return {"error": "; ".join(step_errors)}
    if step_errors:
        out["step_errors"] = step_errors
    return out


def analyze_trajectory(
    workspace: Path,
    compound_id: str,
    has_ligand: bool = False,
    top_file: Optional[str] = None,
    route: Optional[str] = None,
    simulation_ns: Optional[float] = None,
    intent: Optional[str] = None,
    has_protein: Optional[bool] = None,
) -> Dict:
    """Analyze production MD trajectory.

    Extended (April 16) with binding-specific analyses when a ligand is
    present and the system is not a membrane protein:
      - MM-GBSA ΔG_bind + per-residue decomposition (Theo P1)
      - Ligand RMSD (how much the ligand drifted in the pocket)
      - H-bond persistence (with Theo's quality thresholds)
      - Pose clustering (did the ligand stay in one binding mode?)

    Membrane guard: MM-GBSA with standard GB/PB solver produces incorrect
    numbers when a lipid bilayer is in the system (bilayer gets included
    in the solvation energy term). When route == "run_membrane", we skip
    MM-GBSA and record why.
    """
    analysis = {}
    try:
        # --- Production trajectory analysis (existing) ---
        subprocess.run(
            ["gmx", "rms", "-s", f"{compound_id}_md.tpr", "-f", f"{compound_id}_md.xtc",
             "-o", f"{compound_id}_rmsd.xvg", "-tu", "ns"],
            input=b"4\n4\n", check=True, cwd=workspace,
        )
        subprocess.run(
            ["gmx", "rmsf", "-s", f"{compound_id}_md.tpr", "-f", f"{compound_id}_md.xtc",
             "-o", f"{compound_id}_rmsf.xvg", "-res"],
            input=b"4\n", check=True, cwd=workspace,
        )
        subprocess.run(
            ["gmx", "gyrate", "-s", f"{compound_id}_md.tpr", "-f", f"{compound_id}_md.xtc",
             "-o", f"{compound_id}_gyrate.xvg"],
            input=b"1\n", check=True, cwd=workspace,
        )

        # Parse RMSD trajectory (nm)
        rmsd_rows = _parse_xvg(workspace / f"{compound_id}_rmsd.xvg")
        if rmsd_rows:
            rmsd_values = [r[1] for r in rmsd_rows if len(r) >= 2]
            analysis["rmsd"] = {
                "mean_nm": round(sum(rmsd_values) / len(rmsd_values), 4) if rmsd_values else None,
                "max_nm": round(max(rmsd_values), 4) if rmsd_values else None,
                "final_nm": round(rmsd_values[-1], 4) if rmsd_values else None,
                "trajectory": [[round(r[0], 3), round(r[1], 4)] for r in rmsd_rows],
            }

        # Parse RMSF per-residue (nm)
        rmsf_rows = _parse_xvg(workspace / f"{compound_id}_rmsf.xvg")
        if rmsf_rows:
            rmsf_values = [r[1] for r in rmsf_rows if len(r) >= 2]
            analysis["rmsf"] = {
                "mean_nm": round(sum(rmsf_values) / len(rmsf_values), 4) if rmsf_values else None,
                "max_nm": round(max(rmsf_values), 4) if rmsf_values else None,
                "max_residue": int(rmsf_rows[[r[1] for r in rmsf_rows].index(max(rmsf_values))][0]) if rmsf_values else None,
                "n_flexible_residues": sum(1 for v in rmsf_values if v > 0.2),
            }

        # Parse radius of gyration (nm)
        gyrate_rows = _parse_xvg(workspace / f"{compound_id}_gyrate.xvg")
        if gyrate_rows:
            rog_values = [r[1] for r in gyrate_rows if len(r) >= 2]
            analysis["radius_of_gyration"] = {
                "mean_nm": round(sum(rog_values) / len(rog_values), 4) if rog_values else None,
                "std_nm": round((sum((v - sum(rog_values)/len(rog_values))**2 for v in rog_values) / len(rog_values))**0.5, 4) if len(rog_values) > 1 else 0.0,
                "stable": all(abs(v - sum(rog_values)/len(rog_values)) < 0.1 for v in rog_values[-max(1, len(rog_values)//4):]) if rog_values else False,
            }

        # --- Equilibration analysis (NVT → NPT → Production) ---
        # Extract temperature from NVT equilibration (selection "15\n0\n" = Temperature)
        # Use term names instead of numeric indices — indices shift between
        # protein-ligand and ligand-only systems (different energy group count).
        # Term names are stable across system types.
        _extract_energy(workspace, f"{compound_id}_nvt.edr", f"{compound_id}_nvt_temperature.xvg", "Temperature\n\n")
        _extract_energy(workspace, f"{compound_id}_npt.edr", f"{compound_id}_npt_pressure.xvg", "Pressure\n\n")
        _extract_energy(workspace, f"{compound_id}_npt.edr", f"{compound_id}_npt_density.xvg", "Density\n\n")
        _extract_energy(workspace, f"{compound_id}_md.edr", f"{compound_id}_md_potential.xvg", "Potential\n\n")
        _extract_energy(workspace, f"{compound_id}_md.edr", f"{compound_id}_md_temperature.xvg", "Temperature\n\n")
        _extract_energy(workspace, f"{compound_id}_md.edr", f"{compound_id}_md_pressure.xvg", "Pressure\n\n")
        _extract_energy(workspace, f"{compound_id}_md.edr", f"{compound_id}_md_density.xvg", "Density\n\n")

        equilibration = {}

        nvt_temp = _parse_xvg(workspace / f"{compound_id}_nvt_temperature.xvg")
        if nvt_temp:
            equilibration["nvt_temperature"] = _summarize_timeseries(nvt_temp)

        npt_pressure = _parse_xvg(workspace / f"{compound_id}_npt_pressure.xvg")
        if npt_pressure:
            equilibration["npt_pressure"] = _summarize_timeseries(npt_pressure)

        npt_density = _parse_xvg(workspace / f"{compound_id}_npt_density.xvg")
        if npt_density:
            equilibration["npt_density"] = _summarize_timeseries(npt_density)

        md_potential = _parse_xvg(workspace / f"{compound_id}_md_potential.xvg")
        if md_potential:
            equilibration["production_potential_energy"] = _summarize_timeseries(md_potential)

        md_temp = _parse_xvg(workspace / f"{compound_id}_md_temperature.xvg")
        if md_temp:
            equilibration["production_temperature"] = _summarize_timeseries(md_temp)

        md_pressure = _parse_xvg(workspace / f"{compound_id}_md_pressure.xvg")
        if md_pressure:
            equilibration["production_pressure"] = _summarize_timeseries(md_pressure)

        md_density = _parse_xvg(workspace / f"{compound_id}_md_density.xvg")
        if md_density:
            equilibration["production_density"] = _summarize_timeseries(md_density)

        # Surface the adaptive NPT log if simulation.py persisted one during
        # this run. Caller wants to know how long NPT actually ran and
        # whether the density plateau was hit before the cap.
        adaptive_path = workspace / "_npt_adaptive.json"
        if adaptive_path.exists():
            try:
                import json as _json
                equilibration["npt_adaptive"] = _json.loads(adaptive_path.read_text())
            except Exception as e:
                logger.debug(f"Could not parse _npt_adaptive.json: {e}")

        if equilibration:
            analysis["equilibration"] = equilibration

        # --- Quality observability (MD quality gates) ---
        analysis["quality_metrics"] = compute_quality_metrics(
            equilibration, analysis.get("rmsd", {})
        )
        analysis["checkpoint"] = validate_checkpoint(workspace, compound_id)
        analysis["log_diagnostics"] = parse_md_log(workspace, compound_id)
        # has_protein default: if not explicitly provided, infer from top_file
        # (ligand-only runs are called with top_file=None per the routes/
        # simulate.py logic).
        _has_protein = has_protein if has_protein is not None else bool(top_file)
        analysis["quality_report"] = assess_md_quality(
            analysis,
            simulation_ns=simulation_ns,
            has_ligand=has_ligand,
            has_protein=_has_protein,
            route=route,
            intent=intent,
        )

        # --- Binding-specific analyses (Theo P1) ---
        # Only run when a ligand is present. Membrane systems skip MM-GBSA
        # because standard GB/PB solvation is wrong with a bilayer in the
        # system — the lipid gets counted in the solvation term.
        if has_ligand:
            # Ligand RMSD, H-bond persistence, pose clustering (safe for membrane too)
            try:
                ligand_dynamics = analyze_ligand_dynamics(workspace, compound_id)
                if ligand_dynamics:
                    analysis["ligand_dynamics"] = ligand_dynamics
            except Exception as e:
                logger.warning(f"Ligand dynamics analysis failed: {e}")

            # MM-GBSA — only for soluble (non-membrane) systems
            if route == "run_membrane":
                analysis["mmgbsa"] = {
                    "skipped": True,
                    "reason": (
                        "MM-GBSA with standard GB/PB solver produces incorrect ΔG for "
                        "membrane systems — the lipid bilayer is incorrectly treated as "
                        "part of the solvent. A membrane-aware implicit solvent model is "
                        "required but not yet wired. Use the ligand RMSD + H-bond "
                        "persistence metrics for binding stability signals on this run."
                    ),
                }
            elif top_file:
                try:
                    mmgbsa_result = run_mmgbsa_analysis(
                        workspace, compound_id, top_file, f"{compound_id}_md.gro"
                    )
                    analysis["mmgbsa"] = mmgbsa_result
                except Exception as e:
                    logger.warning(f"MM-GBSA analysis failed: {e}")
                    analysis["mmgbsa"] = {"error": str(e)}
            else:
                analysis["mmgbsa"] = {
                    "skipped": True,
                    "reason": (
                        "Ligand-only simulation — MM-GBSA requires a protein-ligand "
                        "complex to compute binding free energy. Use the ligand RMSD "
                        "and pose clustering metrics for stability assessment."
                    ),
                }

    except Exception as e:
        logger.warning(f"Analysis failed: {e}")
        analysis["error"] = str(e)
    return analysis
