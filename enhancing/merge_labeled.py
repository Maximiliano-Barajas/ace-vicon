"""
merge_labeled.py

Takes a folder of labeled per-marker CSVs (one file per body part) and
combines them into a single CSV matching the unlabeled format:
    - 3 header rows
    - columns: Frame, SubFrame, TX, TY, TZ (x14 markers, in body part order)

Usage:
    python merge_labeled.py <folder> <output_csv>

Example:
    python merge_labeled.py firstserve/ firstserve_merged.csv
    python merge_labeled.py serve2/     serve2_merged.csv
"""

import os
import sys
import numpy as np
import pandas as pd

MARKER_ORDER = [
    "head",
    "chest",
    "leftshoulder",
    "rightshoulder",
    "leftelbow",
    "rightelbow",
    "lefthand",
    "righthand",
    "lefthip",
    "righthip",
    "leftknee",
    "rightknee",
    "leftfoot",
    "rightfoot",
]


def read_individual_csv(filepath):
    """Read a single labeled marker CSV and return frames, TX, TY, TZ arrays."""
    raw = pd.read_csv(filepath, header=None, dtype=str)
    data = raw.iloc[3:].reset_index(drop=True)
    frames = pd.to_numeric(data.iloc[:, 0], errors="coerce").values
    tx = pd.to_numeric(data.iloc[:, 2], errors="coerce").values
    ty = pd.to_numeric(data.iloc[:, 3], errors="coerce").values
    tz = pd.to_numeric(data.iloc[:, 4], errors="coerce").values
    return frames, tx, ty, tz


def merge_labeled(folder: str, output_csv: str):
    # Find which markers are present in the folder
    available = {}
    for name in MARKER_ORDER:
        path = os.path.join(folder, f"{name}.csv")
        if os.path.exists(path):
            available[name] = path
        else:
            print(f"  WARNING: {name}.csv not found in {folder} — will be filled with NaN")

    if not available:
        print(f"No marker CSVs found in {folder}. Check the folder path.")
        sys.exit(1)

    # Load all markers and find frame count
    loaded = {}
    frames_ref = None
    for name, path in available.items():
        frames, tx, ty, tz = read_individual_csv(path)
        loaded[name] = (frames, tx, ty, tz)
        if frames_ref is None:
            frames_ref = frames

    n_frames = len(frames_ref)
    print(f"Frames: {n_frames}  Markers found: {len(available)}/{len(MARKER_ORDER)}")

    # Build header rows matching unlabeled format
    row0 = ["Frame", "Sub Frame"]
    row1 = ["", ""]
    row2 = ["Frames", "Frames"]

    for name in MARKER_ORDER:
        row0 += [f"{name}", "", ""]
        row1 += ["TX", "TY", "TZ"]
        row2 += ["mm", "mm", "mm"]

    rows = [row0, row1, row2]

    # Build data rows
    for fi in range(n_frames):
        row = [str(int(frames_ref[fi])), "0"]
        for name in MARKER_ORDER:
            if name in loaded:
                _, tx, ty, tz = loaded[name]
                row.append("" if np.isnan(tx[fi]) else f"{tx[fi]:.3f}")
                row.append("" if np.isnan(ty[fi]) else f"{ty[fi]:.3f}")
                row.append("" if np.isnan(tz[fi]) else f"{tz[fi]:.3f}")
            else:
                row += ["", "", ""]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, header=False)
    print(f"Saved: {output_csv}")
    print(f"Columns: Frame + SubFrame + {len(MARKER_ORDER)} markers x 3 = {2 + len(MARKER_ORDER)*3} total")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_labeled.py <folder> <output_csv>")
        sys.exit(1)

    merge_labeled(sys.argv[1], sys.argv[2])
