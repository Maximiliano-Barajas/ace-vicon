"""
read_serve.py

Reads a raw unmarked motion capture CSV (Motive export format) and returns
the frame numbers and raw 3D marker positions.

No marker labeling is done here — that is handled separately.
Each marker is returned as 'marker_0', 'marker_1', etc., in column order.

Expected CSV structure:
    Row 0: marker track names (every 3 cols starting at col 2)
    Row 1: TX / TY / TZ axis labels
    Row 2: units (mm)
    Row 3+: data — col 0 = frame number, col 1 = sub frame (ignored),
             then groups of 3 cols = TX, TY, TZ per marker

Returns:
    {
        'frames': np.ndarray,          # frame numbers
        'marker_0': {
            'TX': np.ndarray,          # X position in mm
            'TY': np.ndarray,          # Y position in mm
            'TZ': np.ndarray,          # Z position in mm
        },
        'marker_1': { ... },
        ...
    }
"""

import numpy as np
import pandas as pd


def read_serve(filepath: str) -> dict:
    raw = pd.read_csv(filepath, header=None, dtype=str)

    n_cols = raw.shape[1]
    n_markers = (n_cols - 2) // 3

    # Skip the 3 header rows
    data = raw.iloc[3:].reset_index(drop=True)

    frames = pd.to_numeric(data.iloc[:, 0], errors="coerce").values

    result = {"frames": frames}

    for i in range(n_markers):
        col = 2 + i * 3
        result[f"marker_{i}"] = {
            "TX": pd.to_numeric(data.iloc[:, col],     errors="coerce").values,
            "TY": pd.to_numeric(data.iloc[:, col + 1], errors="coerce").values,
            "TZ": pd.to_numeric(data.iloc[:, col + 2], errors="coerce").values,
        }

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python read_serve.py <path_to_csv>")
        sys.exit(1)

    data = read_serve(sys.argv[1])

    n_frames = len(data["frames"])
    n_markers = len(data) - 1  # exclude 'frames' key

    print(f"Frames:  {n_frames}  ({int(data['frames'][0])} → {int(data['frames'][-1])})")
    print(f"Markers: {n_markers}")
    print()

    for key in list(data.keys())[1:]:
        tx = data[key]["TX"]
        valid = int(np.sum(~np.isnan(tx)))
        print(f"  {key}  —  {valid}/{n_frames} valid frames  TX[0]={tx[0]:.3f}")
