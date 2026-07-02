"""
label_from_reference.py  (Program B)

Labels an unlabeled CSV by matching pairwise marker distances
to the reference skeleton learned by build_reference.py (Program A).

Uses frames where both markers in each pair are valid — no need for
all 14 to be visible simultaneously.

Usage:
    python label_from_reference.py <unlabeled_csv> [reference_skeleton.json]

Example:
    python label_from_reference.py data\\14unL.csv reference_skeleton.json
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_serve import read_serve

MARKER_ORDER = [
    "head", "chest",
    "leftshoulder", "rightshoulder",
    "leftelbow", "rightelbow",
    "lefthand", "righthand",
    "lefthip", "righthip",
    "leftknee", "rightknee",
    "leftfoot", "rightfoot",
]

RIGID_PAIRS = [
    ("head",         "chest"),
    ("chest",        "leftshoulder"),
    ("chest",        "rightshoulder"),
    ("leftshoulder", "rightshoulder"),
    ("leftshoulder", "leftelbow"),
    ("leftelbow",    "lefthand"),
    ("rightshoulder","rightelbow"),
    ("rightelbow",   "righthand"),
    ("lefthip",      "righthip"),
    ("chest",        "lefthip"),
    ("chest",        "righthip"),
    ("lefthip",      "leftknee"),
    ("leftknee",     "leftfoot"),
    ("righthip",     "rightknee"),
    ("rightknee",    "rightfoot"),
]


def compute_pairwise_distance(data: dict, key_a: str, key_b: str) -> float:
    """Mean distance between two markers using frames where both are valid."""
    tx_a, ty_a, tz_a = data[key_a]["TX"], data[key_a]["TY"], data[key_a]["TZ"]
    tx_b, ty_b, tz_b = data[key_b]["TX"], data[key_b]["TY"], data[key_b]["TZ"]

    valid = (~np.isnan(tx_a) & ~np.isnan(tx_b) &
             ~np.isnan(ty_a) & ~np.isnan(ty_b) &
             ~np.isnan(tz_a) & ~np.isnan(tz_b))

    if np.sum(valid) < 2:
        return np.nan

    dist = np.sqrt(
        (tx_a[valid] - tx_b[valid])**2 +
        (ty_a[valid] - ty_b[valid])**2 +
        (tz_a[valid] - tz_b[valid])**2
    )
    return float(np.mean(dist))


def build_cost_matrix(data: dict, marker_keys: list, reference: dict) -> np.ndarray:
    """
    Build a cost matrix where cost[i][j] = error when assigning
    marker_keys[j] to MARKER_ORDER[i].

    For each candidate assignment, score it by how well the pairwise
    distances between assigned markers match the reference distances.
    """
    n_parts   = len(MARKER_ORDER)
    n_markers = len(marker_keys)

    # Precompute all pairwise distances between unlabeled markers
    unl_distances = {}
    for a_idx, ma in enumerate(marker_keys):
        for b_idx, mb in enumerate(marker_keys):
            if a_idx >= b_idx:
                continue
            dist = compute_pairwise_distance(data, ma, mb)
            unl_distances[(a_idx, b_idx)] = dist

    # Build cost matrix
    cost = np.zeros((n_parts, n_markers))

    for i, part in enumerate(MARKER_ORDER):
        for j, marker in enumerate(marker_keys):
            # Score this assignment by checking all rigid pairs involving this part
            total_error = 0.0
            pair_count  = 0

            for part_a, part_b in RIGID_PAIRS:
                pair_key = f"{part_a}|{part_b}"
                if reference["pairs"][pair_key]["mean"] is None:
                    continue
                ref_dist = reference["pairs"][pair_key]["mean"]

                # Is this part involved in this pair?
                if part == part_a:
                    other_part = part_b
                elif part == part_b:
                    other_part = part_a
                else:
                    continue

                # For each possible assignment of other_part to a marker,
                # find the best matching marker and compute error
                other_idx = MARKER_ORDER.index(other_part)
                best_error = np.inf

                for k, other_marker in enumerate(marker_keys):
                    if k == j:
                        continue
                    a_idx = min(j, k)
                    b_idx = max(j, k)
                    unl_dist = unl_distances.get((a_idx, b_idx), np.nan)
                    if np.isnan(unl_dist):
                        continue
                    error = abs(unl_dist - ref_dist)
                    if error < best_error:
                        best_error = error

                if best_error < np.inf:
                    total_error += best_error
                    pair_count  += 1

            cost[i][j] = total_error / pair_count if pair_count > 0 else 9999.0

    return cost


def label_from_reference(csv_path: str, reference_path: str) -> dict:
    with open(reference_path, "r") as f:
        reference = json.load(f)

    data = read_serve(csv_path)
    marker_keys = [k for k in data if k.startswith("marker_")]
    n_markers = len(marker_keys)
    n_parts   = len(MARKER_ORDER)

    print(f"Unlabeled markers: {n_markers}")
    print(f"Building cost matrix...")

    cost_matrix = build_cost_matrix(data, marker_keys, reference)

    # Pad if needed
    if n_parts > n_markers:
        pad = np.full((n_parts, n_parts - n_markers), 9999.0)
        cost_matrix = np.hstack([cost_matrix, pad])

    # Hungarian algorithm
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    labels = {}
    print("\n=== Label Assignments ===")
    for r, c in zip(row_ind, col_ind):
        part = MARKER_ORDER[r]
        if c < n_markers:
            marker = marker_keys[c]
            labels[part] = marker
            print(f"  {part:<16} → {marker}  (cost={cost_matrix[r][c]:.1f})")
        else:
            labels[part] = None
            print(f"  {part:<16} → None  ✗ MISSING")

    return labels


def save_labeled_csv(csv_path: str, labels: dict, output_path: str):
    data = read_serve(csv_path)
    frames = data["frames"]
    n_frames = len(frames)
    marker_keys = [k for k in data if k.startswith("marker_")]
    marker_to_part = {v: k for k, v in labels.items() if v is not None}

    row0 = ["Frame", "Sub Frame"]
    row1 = ["", ""]
    row2 = ["Frames", "Frames"]
    for m in marker_keys:
        part = marker_to_part.get(m, m)
        row0 += [part, "", ""]
        row1 += ["TX", "TY", "TZ"]
        row2 += ["mm", "mm", "mm"]

    rows = [row0, row1, row2]
    for fi in range(n_frames):
        row = [str(int(frames[fi])), "0"]
        for m in marker_keys:
            tx = data[m]["TX"][fi]
            ty = data[m]["TY"][fi]
            tz = data[m]["TZ"][fi]
            row.append("" if np.isnan(tx) else f"{tx:.3f}")
            row.append("" if np.isnan(ty) else f"{ty:.3f}")
            row.append("" if np.isnan(tz) else f"{tz:.3f}")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, header=False)
    print(f"\nSaved labeled CSV: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python label_from_reference.py <unlabeled_csv> [reference_skeleton.json]")
        sys.exit(1)

    csv_path = sys.argv[1]
    ref_path = sys.argv[2] if len(sys.argv) > 2 else "reference_skeleton.json"

    if not os.path.exists(ref_path):
        print(f"Reference skeleton not found: {ref_path}")
        print("Run build_reference.py first.")
        sys.exit(1)

    labels = label_from_reference(csv_path, ref_path)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    out  = os.path.join(os.path.dirname(csv_path), f"{base}_labeled.csv")
    save_labeled_csv(csv_path, labels, out)
