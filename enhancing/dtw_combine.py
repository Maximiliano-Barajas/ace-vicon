"""
dtw_combine.py

Combines multiple labeled serve CSVs using per-body-part DTW alignment
to fill in missing marker data (NaN dropouts).

Workflow:
  1. Reads all labeled CSVs from a folder, preserving any leading metadata lines
  2. Picks the serve with the fewest NaNs as the reference
  3. For each body part independently, DTW-warps every other serve
     onto the reference timeline
  4. At each NaN frame in the reference, averages available values
     from the warped serves
  5. If ALL serves are missing a marker at a frame, interpolates linearly
  6. Saves a filled CSV in the same format including the reference metadata

Usage:
    python dtw_combine.py <folder> [output.csv]
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path


# ── CSV I/O ────────────────────────────────────────────────────────────────────

def read_labeled_csv(filepath):
    """
    Read a labeled CSV, dynamically ignoring any top metadata lines
    and saving them so they can be written back out identically.

    Returns:
        part_names: list of str
        frames: np.array of frame numbers
        data: dict { part_name: { TX, TY, TZ } }  (np.arrays, NaN for missing)
    """
    metadata_lines = []
    header_idx = 0
    
    # Scan the file line-by-line to locate where the structural columns begin
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for idx, line in enumerate(f):
            clean_line = line.strip()
            # Stop if we hit the actual data columns header row
            if "Frame," in clean_line or "Frame,Sub" in clean_line:
                header_idx = idx
                break
            metadata_lines.append(clean_line)

    # Read using pandas, skipping any identified leading metadata rows
    raw = pd.read_csv(filepath, header=None, dtype=str, skiprows=header_idx)

    # Parse part names from row 0, every 3 columns starting at col 2
    part_names = []
    col = 2
    while col < raw.shape[1]:
        name = str(raw.iloc[0, col]).strip()
        if name and name.lower() != "nan":
            part_names.append(name)
        col += 3

    data_rows = raw.iloc[3:].reset_index(drop=True)
    frames = pd.to_numeric(data_rows.iloc[:, 0], errors="coerce").values

    result = {}
    for i, name in enumerate(part_names):
        c = 2 + i * 3
        result[name] = {
            "TX": pd.to_numeric(data_rows.iloc[:, c],     errors="coerce").values,
            "TY": pd.to_numeric(data_rows.iloc[:, c + 1], errors="coerce").values,
            "TZ": pd.to_numeric(data_rows.iloc[:, c + 2], errors="coerce").values,
        }

    # Store the extracted metadata block safely inside the result dictionary
    result["__metadata__"] = metadata_lines

    return part_names, frames, result


def write_labeled_csv(filepath, part_names, frames, data):
    """
    Write a labeled CSV in the original format, appending any preserved metadata lines.
    """
    rows = []
    
    # Write metadata back out to the top of the file if it exists
    if "__metadata__" in data:
        for meta_line in data["__metadata__"]:
            rows.append(meta_line.split(","))

    row0 = ["Frame", "Sub Frame"]
    row1 = ["", ""]
    row2 = ["Frames", "Frames"]
    for name in part_names:
        row0 += [name, "", ""]
        row1 += ["TX", "TY", "TZ"]
        row2 += ["mm", "mm", "mm"]

    rows.extend([row0, row1, row2])
    for fi, frame in enumerate(frames):
        row = [str(int(frame)), "0"]
        for name in part_names:
            tx = data[name]["TX"][fi]
            ty = data[name]["TY"][fi]
            tz = data[name]["TZ"][fi]
            row.append("" if np.isnan(tx) else f"{tx:.3f}")
            row.append("" if np.isnan(ty) else f"{ty:.3f}")
            row.append("" if np.isnan(tz) else f"{tz:.3f}")
        rows.append(row)

    pd.DataFrame(rows).to_csv(filepath, index=False, header=False)
    print(f"Saved: {filepath}")


# ── HELPERS ────────────────────────────────────────────────────────────────────

def nan_fraction(data, part_names):
    """Return fraction of NaN values across all parts and axes."""
    total = valid = 0
    for name in part_names:
        for ax in ("TX", "TY", "TZ"):
            arr = data[name][ax]
            total += len(arr)
            valid += int(np.sum(~np.isnan(arr)))
    return 1.0 - valid / max(total, 1)


def linear_interpolate(arr):
    """
    Linearly interpolate NaN gaps in a 1D array.
    Extrapolates at edges using nearest valid value.
    """
    out = arr.copy()
    n = len(out)
    nans = np.isnan(out)
    if not nans.any():
        return out
    if nans.all():
        return out  # nothing to do

    idx = np.arange(n)
    valid_idx = idx[~nans]
    out[nans] = np.interp(idx[nans], valid_idx, out[valid_idx])
    return out


# ── DTW ───────────────────────────────────────────────────────────────────────

def dtw_warp_path(ref_seq, src_seq):
    """
    Compute DTW warp path between two 1D sequences (NaNs filled before DTW).
    Returns warp_path: list of (ref_idx, src_idx) pairs.
    """
    r = linear_interpolate(ref_seq.copy())
    s = linear_interpolate(src_seq.copy())

    try:
        from dtaidistance import dtw as dtaidtw
        _, paths = dtaidtw.warping_paths(r, s)
        from dtaidistance import dtw_visualisation as dtwvis
        path = dtaidtw.best_path(paths)
        return path
    except ImportError:
        pass

    # Pure-numpy fallback DTW
    N, M = len(r), len(s)
    cost = np.full((N, M), np.inf)
    cost[0, 0] = abs(r[0] - s[0])
    for i in range(1, N):
        cost[i, 0] = cost[i-1, 0] + abs(r[i] - s[0])
    for j in range(1, M):
        cost[0, j] = cost[0, j-1] + abs(r[0] - s[j])
    for i in range(1, N):
        for j in range(1, M):
            cost[i, j] = abs(r[i] - s[j]) + min(cost[i-1, j], cost[i, j-1], cost[i-1, j-1])

    # Traceback
    path = []
    i, j = N - 1, M - 1
    path.append((i, j))
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            move = np.argmin([cost[i-1, j-1], cost[i-1, j], cost[i, j-1]])
            if move == 0:
                i -= 1; j -= 1
            elif move == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))
    path.reverse()
    return path


def warp_series(src_seq, path, ref_len):
    """
    Project src_seq onto the reference timeline using a DTW warp path.
    """
    buckets = [[] for _ in range(ref_len)]
    for ri, si in path:
        buckets[ri].append(src_seq[si])

    out = np.full(ref_len, np.nan)
    for ri, vals in enumerate(buckets):
        valid = [v for v in vals if not np.isnan(v)]
        if valid:
            out[ri] = np.mean(valid)
    return out


# ── MAIN LOGIC ────────────────────────────────────────────────────────────────

def combine_serves(folder, output_path):
    folder = Path(folder)
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {folder}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"  {f.name}")
    print()

    # Load all serves
    serves = []
    for f in csv_files:
        try:
            part_names, frames, data = read_labeled_csv(f)
            serves.append({
                "name": f.name,
                "part_names": part_names,
                "frames": frames,
                "data": data,
                "nan_frac": nan_fraction(data, part_names),
            })
            print(f"  Loaded {f.name}: {len(frames)} frames, "
                  f"{len(part_names)} parts, "
                  f"{serves[-1]['nan_frac']*100:.1f}% NaN")
        except Exception as e:
            print(f"  WARNING: Could not load {f.name}: {e}")

    if not serves:
        print("No valid serves loaded.")
        sys.exit(1)

    # Pick reference: fewest NaNs
    ref = min(serves, key=lambda s: s["nan_frac"])
    others = [s for s in serves if s is not ref]
    print(f"\nReference serve: {ref['name']} ({ref['nan_frac']*100:.1f}% NaN)")
    print(f"Aligning {len(others)} other serve(s) via DTW...\n")

    # Union of all part names (use reference order first, then any extras)
    all_parts = list(ref["part_names"])
    for s in others:
        for p in s["part_names"]:
            if p not in all_parts:
                all_parts.append(p)

    ref_len = len(ref["frames"])
    filled = {}  # part -> {TX, TY, TZ}

    for part in all_parts:
        print(f"  Processing: {part}")

        # Reference sequences for this part
        if part in ref["data"]:
            ref_tx = ref["data"][part]["TX"].copy()
            ref_ty = ref["data"][part]["TY"].copy()
            ref_tz = ref["data"][part]["TZ"].copy()
        else:
            ref_tx = np.full(ref_len, np.nan)
            ref_ty = np.full(ref_len, np.nan)
            ref_tz = np.full(ref_len, np.nan)

        # Identify NaN frames in reference for this part
        ref_nan_mask = np.isnan(ref_tx) | np.isnan(ref_ty) | np.isnan(ref_tz)

        if not ref_nan_mask.any():
            # No gaps — keep reference as-is
            filled[part] = {"TX": ref_tx, "TY": ref_ty, "TZ": ref_tz}
            continue

        # Collect warped values from other serves at NaN frames
        warped_tx_list = []
        warped_ty_list = []
        warped_tz_list = []

        for s in others:
            if part not in s["data"]:
                continue

            src_tx = s["data"][part]["TX"]
            src_ty = s["data"][part]["TY"]
            src_tz = s["data"][part]["TZ"]

            # DTW warp path based on 3D signal magnitude
            ref_mag = np.sqrt(
                np.where(np.isnan(ref_tx), 0, ref_tx)**2 +
                np.where(np.isnan(ref_ty), 0, ref_ty)**2 +
                np.where(np.isnan(ref_tz), 0, ref_tz)**2
            )
            src_mag = np.sqrt(
                np.where(np.isnan(src_tx), 0, src_tx)**2 +
                np.where(np.isnan(src_ty), 0, src_ty)**2 +
                np.where(np.isnan(src_tz), 0, src_tz)**2
            )

            path = dtw_warp_path(ref_mag, src_mag)

            warped_tx = warp_series(src_tx, path, ref_len)
            warped_ty = warp_series(src_ty, path, ref_len)
            warped_tz = warp_series(src_tz, path, ref_len)

            warped_tx_list.append(warped_tx)
            warped_ty_list.append(warped_ty)
            warped_tz_list.append(warped_tz)

        # Merge: start from reference, fill NaNs with average of warped serves
        out_tx = ref_tx.copy()
        out_ty = ref_ty.copy()
        out_tz = ref_tz.copy()

        for fi in np.where(ref_nan_mask)[0]:
            vals_x = [w[fi] for w in warped_tx_list if not np.isnan(w[fi])]
            vals_y = [w[fi] for w in warped_ty_list if not np.isnan(w[fi])]
            vals_z = [w[fi] for w in warped_tz_list if not np.isnan(w[fi])]

            if vals_x:
                out_tx[fi] = np.mean(vals_x)
            if vals_y:
                out_ty[fi] = np.mean(vals_y)
            if vals_z:
                out_tz[fi] = np.mean(vals_z)

        # Final pass: linearly interpolate any remaining NaNs (all serves missing)
        out_tx = linear_interpolate(out_tx)
        out_ty = linear_interpolate(out_ty)
        out_tz = linear_interpolate(out_tz)

        filled[part] = {"TX": out_tx, "TY": out_ty, "TZ": out_tz}

    # Forward the saved reference metadata to the output writer mapping
    if "__metadata__" in ref["data"]:
        filled["__metadata__"] = ref["data"]["__metadata__"]

    # Report
    total_ref_nans = sum(
        int(np.sum(np.isnan(ref["data"].get(p, {}).get("TX", np.array([np.nan])))))
        for p in all_parts
    )
    total_filled_nans = sum(
        int(np.sum(np.isnan(filled[p]["TX"]))) for p in all_parts
    )
    print(f"\nReference NaN frames (TX axis): {total_ref_nans}")
    print(f"Remaining NaN frames after fill: {total_filled_nans}")
    print()

    write_labeled_csv(output_path, all_parts, ref["frames"], filled)
    print(f"\nDone! View with:")
    print(f"  python plot_serve.py M {output_path}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python dtw_combine.py <folder> [output.csv]")
        sys.exit(1)

    folder = sys.argv[1]
    if not os.path.isdir(folder):
        print(f"Not a directory: {folder}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        output = sys.argv[2]
    else:
        output = os.path.join(folder, "combined_serve.csv")

    combine_serves(folder, output)