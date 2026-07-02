"""
ACE Tennis Serve Grader — Elbow Angle, Phase-Based
----------------------------------------------------
File location: src/tennis_serve_grader.py

Compares a customer's elbow angle at the start frame of each of the
8 segmentation phases against the DTW barycenter (pro reference).

The start frame of each phase is used as the "trophy position" for
that phase — each phase boundary is anchored to a key biomechanical
event by Max's segmentation pipeline (maximum_knee_bend,
maximum_shoulder_external_rotation, etc.), making the start frame
the most biomechanically significant moment in the phase.

MARKER_ORDER column mapping (from dtw/constants.py):
    right_shoulder = Marker 6 → columns 15, 16, 17
    right_elbow    = Marker 3 → columns  6,  7,  8
    right_hand     = Marker 8 → columns 21, 22, 23

Pipeline:
    barycenter2.npy (pro reference)
        + SegmentationResult.phases (Max)
        + customer angle array (Biplav/Devyn)
        → grade_serve()
        → print_report()
"""

from __future__ import annotations

import os
import numpy as np
from segmentation.result import PHASE_NAMES

# ─────────────────────────────────────────────
# 1. MARKER COLUMN INDICES
#    Derived from MARKER_ORDER in dtw/constants.py
#    Each marker occupies 3 consecutive columns (TX, TY, TZ)
# ─────────────────────────────────────────────

# marker index (0-based) × 3 = first column
RSHO_COLS = slice(15, 18)   # right_shoulder (marker 6)
RELB_COLS = slice(6,  9)    # right_elbow    (marker 3)
RHAN_COLS = slice(21, 24)   # right_hand     (marker 8)

# ─────────────────────────────────────────────
# 2. BARYCENTER PATH
#    Points to barycenter2.npy — DBA with dtw-python on individual/ serves
#    Override by passing pro_barycenter= to grade_serve()
# ─────────────────────────────────────────────

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BARYCENTER = os.path.join(_HERE, "..", "dtw", "barycenter2.npy")

# ─────────────────────────────────────────────
# 3. PHASE WEIGHTS
#    Cocking / Acceleration / Contact weighted highest —
#    these phases contain the power-generating and ball-strike events.
# ─────────────────────────────────────────────

PHASE_WEIGHTS = {
    "Start_Stance":  0.05,
    "Release":       0.10,
    "Loading":       0.15,
    "Cocking":       0.20,
    "Acceleration":  0.20,
    "Contact":       0.20,
    "Deceleration":  0.05,
    "Finish":        0.05,
}

# ─────────────────────────────────────────────
# 4. SCORING THRESHOLDS (degrees difference)
# ─────────────────────────────────────────────

ANGLE_THRESHOLDS = [
    (5,   95, "Excellent"),
    (15,  80, "Good"),
    (30,  60, "Fair"),
    (999, 35, "Poor"),
]

TIER_ICON = {"Excellent": "✓", "Good": "~", "Fair": "!", "Poor": "✗"}

# ─────────────────────────────────────────────
# 5. PHASE-SPECIFIC FEEDBACK
# ─────────────────────────────────────────────

PHASE_CONTEXT = {
    "Start_Stance":  "your starting stance and initial arm position",
    "Release":       "your toss release — elbow should begin to rise",
    "Loading":       "your loading phase — elbow pulls back as knees bend",
    "Cocking":       "the trophy/cocking position — this is the key power-loading moment",
    "Acceleration":  "your acceleration phase — elbow drives forward toward the ball",
    "Contact":       "ball contact — elbow should be near full extension",
    "Deceleration":  "your deceleration — controlled arm slowdown after contact",
    "Finish":        "your follow-through and finish position",
}

FEEDBACK_TEMPLATES = {
    "Excellent": "Elbow angle matches the pro at {phase_label} — excellent position.",
    "Good":      "Elbow is slightly off at {phase_label}. Minor adjustment needed for {context}.",
    "Fair":      "Elbow angle deviates noticeably at {phase_label}. Focus on {context}.",
    "Poor":      "Elbow is significantly misaligned at {phase_label}. Prioritise drills for {context}.",
}

# ─────────────────────────────────────────────
# 6. BIOMECHANICS HELPERS
# ─────────────────────────────────────────────

def compute_elbow_angle(frame: np.ndarray) -> float:
    """
    Compute the elbow angle (degrees) from a single barycenter frame.

    Uses the right_shoulder → right_elbow → right_hand triplet.
    The angle is computed at the elbow joint (vertex = right_elbow).

    Args:
        frame: 1D array of length 42 (one row of the barycenter)

    Returns:
        elbow angle in degrees
    """
    shoulder = frame[RSHO_COLS]
    elbow    = frame[RELB_COLS]
    hand     = frame[RHAN_COLS]

    # Vectors from elbow to shoulder and elbow to hand
    v1 = shoulder - elbow
    v2 = hand     - elbow

    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def score_angle(diff_deg: float) -> tuple:
    for threshold, points, tier in ANGLE_THRESHOLDS:
        if diff_deg < threshold:
            return points, tier
    return 35, "Poor"


# ─────────────────────────────────────────────
# 7. MAIN GRADING FUNCTION
# ─────────────────────────────────────────────

def grade_serve(
    phase_start_frames: dict[str, int],
    customer_elbow_angles: dict[str, float],
    pro_barycenter: np.ndarray | str | None = None,
) -> dict:
    """
    Grade a customer's serve elbow angle at each phase's start frame.

    Parameters
    ----------
    phase_start_frames : dict
        Maps phase name → start frame INDEX (not Vicon frame number).
        Comes from Max's SegmentationResult — use phase_to_index_range()
        to convert Vicon frames to indices first.
        Example: {"Start_Stance": 0, "Release": 45, "Loading": 112, ...}

    customer_elbow_angles : dict
        Maps phase name → elbow angle (degrees) at that phase's start frame.
        Computed by Biplav/Devyn from the customer's aligned marker data.
        Example: {"Start_Stance": 168.0, "Release": 155.0, ...}

    pro_barycenter : np.ndarray or str or None
        The DTW barycenter array (n_frames, 42), a path to a .npy file,
        or None to use the default barycenter2.npy.

    Returns
    -------
    dict with per-phase results, overall score, and overall grade
    """
    # Load barycenter if needed
    if pro_barycenter is None:
        pro_barycenter = np.load(DEFAULT_BARYCENTER)
    elif isinstance(pro_barycenter, str):
        pro_barycenter = np.load(pro_barycenter)

    results = {"phases": {}}
    weighted_total = 0.0
    weight_used    = 0.0

    for phase in PHASE_NAMES:
        start_idx      = phase_start_frames.get(phase)
        customer_angle = customer_elbow_angles.get(phase)

        if start_idx is None or customer_angle is None:
            results["phases"][phase] = {"phase_score": None, "summary": f"{phase} — no data"}
            continue

        # Extract pro elbow angle from barycenter at the phase start frame
        start_idx = min(start_idx, len(pro_barycenter) - 1)
        pro_angle = compute_elbow_angle(pro_barycenter[start_idx])

        diff  = abs(customer_angle - pro_angle)
        score, tier = score_angle(diff)

        phase_label = phase.replace("_", " ")
        feedback = FEEDBACK_TEMPLATES[tier].format(
            phase_label=phase_label,
            context=PHASE_CONTEXT[phase],
        )

        results["phases"][phase] = {
            "customer_angle": round(customer_angle, 1),
            "pro_angle":      round(pro_angle, 1),
            "diff":           round(diff, 1),
            "score":          score,
            "tier":           tier,
            "feedback":       feedback,
        }

        w = PHASE_WEIGHTS[phase]
        weighted_total += score * w
        weight_used    += w

    if weight_used == 0:
        results["overall_score"] = None
        results["overall_grade"] = "No data provided."
        return results

    overall = round(weighted_total / weight_used, 1)
    results["overall_score"] = overall

    if overall >= 90:
        results["overall_grade"] = "A — Pro-level serve"
    elif overall >= 75:
        results["overall_grade"] = "B — Strong serve, minor adjustments needed"
    elif overall >= 60:
        results["overall_grade"] = "C — Developing serve, focused practice recommended"
    else:
        results["overall_grade"] = "D — Fundamentals need significant work"

    return results


# ─────────────────────────────────────────────
# 8. REPORT PRINTER
# ─────────────────────────────────────────────

def print_report(results: dict) -> None:
    sep = "─" * 64
    print(f"\n{'ACE TENNIS SERVE — ELBOW ANGLE REPORT':^64}")
    print(sep)

    for phase in PHASE_NAMES:
        p = results["phases"].get(phase)
        if not p or p.get("score") is None:
            print(f"\n  {phase.replace('_', ' ')}  — no data")
            continue

        icon       = TIER_ICON[p["tier"]]
        weight_pct = int(PHASE_WEIGHTS[phase] * 100)
        print(f"\n  {phase.replace('_', ' ')}  ({weight_pct}% of grade)")
        print(f"  [{icon}] {p['tier']:9s}  Score: {p['score']:3d}/100"
              f"   Customer: {p['customer_angle']}°  Pro: {p['pro_angle']}°  Diff: {p['diff']}°")
        print(f"      → {p['feedback']}")

    print(f"\n{sep}")
    print(f"  OVERALL SCORE : {results['overall_score']}/100")
    print(f"  GRADE         : {results['overall_grade']}")
    print(sep)

    
def align_to_barycenter(customer_array, barycenter):
    """
    Align a customer marker array to the barycenter timeline using DTW.

    The customer serve may be longer or shorter than the barycenter.
    DTW finds the optimal warping path, then we resample the customer
    frames onto the barycenter frame count so phase start indices are
    directly comparable between customer and pro.

    Parameters
    ----------
    customer_array : np.ndarray  shape (n_customer_frames, 42)
    barycenter     : np.ndarray  shape (n_barycenter_frames, 42)

    Returns
    -------
    np.ndarray shape (n_barycenter_frames, 42)
    """
    from dtw import dtw
    from scipy.spatial.distance import cdist

    print("Aligning customer serve to pro barycenter via DTW...")
    dist_mat  = cdist(customer_array, barycenter)
    alignment = dtw(dist_mat, distance_only=False)

    n_bc  = barycenter.shape[0]
    accum = [[] for _ in range(n_bc)]
    for cust_idx, bc_idx in zip(alignment.index1, alignment.index2):
        accum[bc_idx].append(customer_array[cust_idx])

    aligned = np.zeros_like(barycenter)
    for j in range(n_bc):
        if accum[j]:
            aligned[j] = np.mean(accum[j], axis=0)
        else:
            aligned[j] = customer_array[
                alignment.index1[np.argmin(np.abs(alignment.index2 - j))]
            ]

    print(f"  Customer: {customer_array.shape[0]} frames -> aligned to {n_bc} frames")
    return aligned


def run_from_csv(serve_folder: str) -> dict:
    """
    Run the full grading pipeline on a real customer Vicon CSV folder.

    Pipeline:
        1. Load + clean the CSV (same process as pro serves)
        2. DTW-align customer timeline to barycenter
        3. Segment the serve into 8 phases
        4. Extract elbow angle at each phase start frame
        5. Grade against the pro barycenter
    """
    from segmentation.io import load_serve_from_folder
    from segmentation.pipeline import segment_serve
    from segmentation.result import phase_to_index_range
    from prepare_data import interpolate_nans, filter_nan_frames, normalize_serve, convert

    # Step 1: Load and clean
    print(f"\nLoading customer serve from: {serve_folder}")
    serve = load_serve_from_folder(serve_folder)
    serve = filter_nan_frames(serve)
    serve = interpolate_nans(serve)
    serve = normalize_serve(serve)

    # Step 2: DTW align to barycenter
    barycenter    = np.load(DEFAULT_BARYCENTER)
    marker_array  = convert(serve)
    aligned_array = align_to_barycenter(marker_array, barycenter)

    # Step 3: Segment (runs on original serve for signal quality)
    print("Running segmentation...")
    seg_result = segment_serve(serve)
    if seg_result.warnings:
        for w in seg_result.warnings:
            print(f"  WARNING: {w}")
    frames = seg_result.frames

    # Step 4: Extract elbow angle at each phase start frame
    phase_start_frames    = {}
    customer_elbow_angles = {}
    print("\nPhase start frames detected:")
    for phase in PHASE_NAMES:
        if phase not in seg_result.phases:
            print(f"  {phase}: not found in segmentation")
            continue
        vicon_start, vicon_end = seg_result.phases[phase]
        start_idx, _ = phase_to_index_range(frames, (vicon_start, vicon_end))
        start_idx = min(start_idx, len(aligned_array) - 1)
        elbow_angle = compute_elbow_angle(aligned_array[start_idx])
        phase_start_frames[phase]    = start_idx
        customer_elbow_angles[phase] = elbow_angle
        print(f"  {phase:15s} start_idx={start_idx:4d}  elbow={elbow_angle:.1f}")

    # Step 5: Grade
    return grade_serve(phase_start_frames, customer_elbow_angles, barycenter)


def run_dummy() -> dict:
    """Run grader with dummy data for testing without a real CSV."""
    dummy_phase_starts = {
        "Start_Stance": 0,   "Release": 45,  "Loading": 112,
        "Cocking": 140,      "Acceleration": 160, "Contact": 195,
        "Deceleration": 196, "Finish": 240,
    }
    dummy_customer_angles = {
        "Start_Stance": 168.0, "Release": 155.0, "Loading": 125.0,
        "Cocking": 110.0,      "Acceleration": 128.0, "Contact": 172.0,
        "Deceleration": 145.0, "Finish": 95.0,
    }
    return grade_serve(dummy_phase_starts, dummy_customer_angles)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ACE Tennis Serve Grader")
    parser.add_argument(
        "--csv", metavar="FOLDER", default=None,
        help="Path to customer Vicon serve folder. Omit to use dummy data.",
    )
    args = parser.parse_args()

    if args.csv:
        results = run_from_csv(args.csv)
    else:
        print("No CSV provided — running with dummy data.")
        results = run_dummy()

    print_report(results)