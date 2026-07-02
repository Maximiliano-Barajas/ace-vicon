"""
build_reference.py  (Program A)

Learns a reference skeleton from multiple labeled merged CSVs.
Computes mean pairwise distances between all body part combinations,
focusing on rigid bone connections that stay constant regardless of pose.

Output: reference_skeleton.json

Usage:
    python build_reference.py <merged_csv1> <merged_csv2> ...

Example:
    python build_reference.py data\\firstserve_merged.csv data\\second_merged.csv
"""

import sys
import os
import json
import numpy as np
from itertools import combinations

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

# Rigid bone pairs — distances that stay constant regardless of pose
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


def compute_pairwise_distance(data: dict, key_a: str, key_b: str) -> tuple[float, float]:
    """
    Compute mean and std of distance between two markers
    using only frames where both are valid.
    """
    tx_a, ty_a, tz_a = data[key_a]["TX"], data[key_a]["TY"], data[key_a]["TZ"]
    tx_b, ty_b, tz_b = data[key_b]["TX"], data[key_b]["TY"], data[key_b]["TZ"]

    valid = (~np.isnan(tx_a) & ~np.isnan(tx_b) &
             ~np.isnan(ty_a) & ~np.isnan(ty_b) &
             ~np.isnan(tz_a) & ~np.isnan(tz_b))

    if np.sum(valid) < 2:
        return np.nan, np.nan

    dist = np.sqrt(
        (tx_a[valid] - tx_b[valid])**2 +
        (ty_a[valid] - ty_b[valid])**2 +
        (tz_a[valid] - tz_b[valid])**2
    )
    return float(np.mean(dist)), float(np.std(dist))


def build_reference(csv_files: list, output_path: str = "reference_skeleton.json"):
    all_distances = {f"{a}|{b}": [] for a, b in RIGID_PAIRS}

    for path in csv_files:
        print(f"Loading: {path}")
        data = read_serve(path)

        for a, b in RIGID_PAIRS:
            i = MARKER_ORDER.index(a)
            j = MARKER_ORDER.index(b)
            key_a = f"marker_{i}"
            key_b = f"marker_{j}"
            mean_dist, std_dist = compute_pairwise_distance(data, key_a, key_b)
            if not np.isnan(mean_dist):
                all_distances[f"{a}|{b}"].append(mean_dist)
                print(f"  {a:<16} ↔ {b:<16} dist={mean_dist:.1f}mm  std={std_dist:.1f}mm")

    # Average across all serves
    reference = {"pairs": {}, "marker_order": MARKER_ORDER, "rigid_pairs": RIGID_PAIRS}
    print("\n=== Reference Pairwise Distances ===")
    for pair_key, dists in all_distances.items():
        if len(dists) > 0:
            mean = float(np.mean(dists))
            std  = float(np.std(dists))
            reference["pairs"][pair_key] = {"mean": mean, "std": std, "n": len(dists)}
            print(f"  {pair_key:<35} mean={mean:.1f}mm  std={std:.1f}mm")
        else:
            reference["pairs"][pair_key] = {"mean": None, "std": None, "n": 0}
            print(f"  {pair_key:<35} NO DATA")

    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"\nSaved reference skeleton to: {output_path}")
    print(f"Built from {len(csv_files)} serves.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_reference.py <merged_csv1> <merged_csv2> ...")
        sys.exit(1)

    build_reference(sys.argv[1:])
