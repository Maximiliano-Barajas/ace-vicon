"""
label_markers.py

Assigns anatomical labels to raw unlabeled markers using biomechanical rules.

Sequential logic — each step cements its assignment before the next:
    1.  head           — highest in Z
    2.  feet           — lowest 2 in Z
    3.  knees          — most constant distance to feet
    4.  racket hand    — greatest Y range
    5.  shoulders      — 2 closest to head by vector distance
    6.  chest          — closest to expected position (below head, between shoulders)
    7.  racket elbow   — most constant distance to racket hand (from remaining)
    8.  racket shoulder — shoulder with smallest average distance to racket elbow
    9.  other shoulder — the remaining shoulder
    10. other elbow    — most constant distance to other shoulder (from remaining)
    11. other hand     — most constant distance to other elbow (from remaining)
    12. hip_a, hip_b   — last 2 remaining
"""

from __future__ import annotations
import numpy as np

BODY_PARTS = [
    "head",
    "chest",
    "shoulder_a",
    "shoulder_b",
    "racket_shoulder",
    "other_shoulder",
    "racket_elbow",
    "other_elbow",
    "racket_hand",
    "other_hand",
    "hip_a",
    "hip_b",
    "knee_a",
    "knee_b",
    "foot_a",
    "foot_b",
]


def _mean_position(data: dict, marker: str) -> tuple[float, float, float]:
    tx = data[marker]["TX"]
    ty = data[marker]["TY"]
    tz = data[marker]["TZ"]
    valid = ~(np.isnan(tx) | np.isnan(ty) | np.isnan(tz))
    if not np.any(valid):
        return np.nan, np.nan, np.nan
    return float(np.nanmean(tx[valid])), float(np.nanmean(ty[valid])), float(np.nanmean(tz[valid]))


def _mean_z(data: dict, marker: str) -> float:
    return float(np.nanmean(data[marker]["TZ"]))


def _y_range(data: dict, marker: str) -> float:
    ty = data[marker]["TY"]
    valid = ty[~np.isnan(ty)]
    if len(valid) == 0:
        return 0.0
    return float(valid.max() - valid.min())


def _distance_variance(data: dict, m1: str, m2: str) -> float:
    """Variance of frame-by-frame distance — lower = more rigid."""
    tx1, ty1, tz1 = data[m1]["TX"], data[m1]["TY"], data[m1]["TZ"]
    tx2, ty2, tz2 = data[m2]["TX"], data[m2]["TY"], data[m2]["TZ"]
    dist = np.sqrt((tx1-tx2)**2 + (ty1-ty2)**2 + (tz1-tz2)**2)
    valid = ~np.isnan(dist)
    if np.sum(valid) < 2:
        return np.inf
    return float(np.var(dist[valid]))


def _mean_vector_distance(data: dict, m1: str, m2: str) -> float:
    """Mean vector distance between two markers using their mean positions."""
    pos1 = np.array(_mean_position(data, m1))
    pos2 = np.array(_mean_position(data, m2))
    if np.any(np.isnan(pos1)) or np.any(np.isnan(pos2)):
        return np.inf
    return float(np.linalg.norm(pos1 - pos2))


def _most_constant_distance(data: dict, anchor: str, candidates: list[str]) -> str:
    """Return candidate with lowest distance variance to anchor."""
    return min(candidates, key=lambda m: _distance_variance(data, anchor, m))


def _closest_to(data: dict, anchor: str, candidates: list[str]) -> str:
    """Return candidate with smallest mean vector distance to anchor."""
    return min(candidates, key=lambda m: _mean_vector_distance(data, anchor, m))


def label_markers(data: dict) -> dict:
    warnings: list[str] = []
    marker_keys = [k for k in data if k.startswith("marker_")]
    unassigned = list(marker_keys)
    labels: dict[str, str | None] = {part: None for part in BODY_PARTS}

    def assign(part: str, marker: str):
        labels[part] = marker
        if marker in unassigned:
            unassigned.remove(marker)

    # ── 1. HEAD — highest in Z ────────────────────────────────────────────
    head = max(unassigned, key=lambda m: _mean_z(data, m))
    assign("head", head)
    print(f"  1. head:            {labels['head']} (Z={_mean_z(data, labels['head']):.1f})")

    # ── 2. FEET — lowest 2 in Z ───────────────────────────────────────────
    sorted_by_z = sorted(unassigned, key=lambda m: _mean_z(data, m))
    assign("foot_a", sorted_by_z[0])
    assign("foot_b", sorted_by_z[1])
    print(f"  2. foot_a:          {labels['foot_a']} (Z={_mean_z(data, labels['foot_a']):.1f})")
    print(f"     foot_b:          {labels['foot_b']} (Z={_mean_z(data, labels['foot_b']):.1f})")

    # ── 3. KNEES — most constant distance to each foot ────────────────────
    knee_a = _most_constant_distance(data, labels["foot_a"], unassigned)
    assign("knee_a", knee_a)
    knee_b = _most_constant_distance(data, labels["foot_b"], unassigned)
    assign("knee_b", knee_b)
    print(f"  3. knee_a:          {labels['knee_a']} (var={_distance_variance(data, labels['foot_a'], labels['knee_a']):.1f})")
    print(f"     knee_b:          {labels['knee_b']} (var={_distance_variance(data, labels['foot_b'], labels['knee_b']):.1f})")

    # ── 4. RACKET HAND — greatest Y range ─────────────────────────────────
    racket_hand = max(unassigned, key=lambda m: _y_range(data, m))
    assign("racket_hand", racket_hand)
    print(f"  4. racket hand:     {labels['racket_hand']} (Y range={_y_range(data, labels['racket_hand']):.1f})")

    # ── 5. SHOULDERS — 2 closest to head ──────────────────────────────────
    s_a = _closest_to(data, labels["head"], unassigned)
    assign("shoulder_a", s_a)
    s_b = _closest_to(data, labels["head"], unassigned)
    assign("shoulder_b", s_b)
    print(f"  5. shoulder_a:      {labels['shoulder_a']} (dist to head={_mean_vector_distance(data, labels['head'], labels['shoulder_a']):.1f})")
    print(f"     shoulder_b:      {labels['shoulder_b']} (dist to head={_mean_vector_distance(data, labels['head'], labels['shoulder_b']):.1f})")

    # ── 6. CHEST — closest to expected position (below head, between shoulders) ──
    head_pos   = np.array(_mean_position(data, labels["head"]))
    sa_pos     = np.array(_mean_position(data, labels["shoulder_a"]))
    sb_pos     = np.array(_mean_position(data, labels["shoulder_b"]))
    shoulder_mid = (sa_pos + sb_pos) / 2
    # Expected chest = midpoint of shoulders, dropped down by head-to-shoulder distance
    head_to_shoulder_dist = float(np.linalg.norm(head_pos - shoulder_mid))
    expected_chest = shoulder_mid.copy()
    expected_chest[2] -= head_to_shoulder_dist  # drop down in Z

    def dist_to_expected_chest(m):
        pos = np.array(_mean_position(data, m))
        return float(np.linalg.norm(pos - expected_chest))

    chest = min(unassigned, key=dist_to_expected_chest)
    assign("chest", chest)
    print(f"  6. chest:           {labels['chest']} (dist to expected={dist_to_expected_chest(labels['chest']):.1f})")

    # ── 7. RACKET ELBOW — most constant distance to racket hand ───────────
    racket_elbow = _most_constant_distance(data, labels["racket_hand"], unassigned)
    assign("racket_elbow", racket_elbow)
    print(f"  7. racket elbow:    {labels['racket_elbow']} (var={_distance_variance(data, labels['racket_hand'], labels['racket_elbow']):.1f})")

    # ── 8. RACKET SHOULDER — shoulder with smallest avg distance to racket elbow ─
    dist_sa = _mean_vector_distance(data, labels["racket_elbow"], labels["shoulder_a"])
    dist_sb = _mean_vector_distance(data, labels["racket_elbow"], labels["shoulder_b"])
    if dist_sa <= dist_sb:
        assign("racket_shoulder", labels["shoulder_a"])
        assign("other_shoulder",  labels["shoulder_b"])
    else:
        assign("racket_shoulder", labels["shoulder_b"])
        assign("other_shoulder",  labels["shoulder_a"])
    print(f"  8. racket shoulder: {labels['racket_shoulder']} (dist to racket elbow={min(dist_sa, dist_sb):.1f})")
    print(f"  9. other shoulder:  {labels['other_shoulder']} (dist to racket elbow={max(dist_sa, dist_sb):.1f})")

    # ── 10. OTHER ELBOW — most constant distance to other shoulder ─────────
    other_elbow = _most_constant_distance(data, labels["other_shoulder"], unassigned)
    assign("other_elbow", other_elbow)
    print(f" 10. other elbow:     {labels['other_elbow']} (var={_distance_variance(data, labels['other_shoulder'], labels['other_elbow']):.1f})")

    # ── 11. OTHER HAND — most constant distance to other elbow ────────────
    other_hand = _most_constant_distance(data, labels["other_elbow"], unassigned)
    assign("other_hand", other_hand)
    print(f" 11. other hand:      {labels['other_hand']} (var={_distance_variance(data, labels['other_elbow'], labels['other_hand']):.1f})")

    # ── 12. HIPS — last 2 remaining ───────────────────────────────────────
    if len(unassigned) >= 2:
        hip_a, hip_b = unassigned[0], unassigned[1]
        assign("hip_a", hip_a)
        assign("hip_b", hip_b)
        if len(unassigned) > 0:
            warnings.append(f"Extra unassigned markers after hips: {unassigned}")
        print(f" 12. hip_a:           {labels['hip_a']}")
        print(f"     hip_b:           {labels['hip_b']}")
    elif len(unassigned) == 1:
        assign("hip_a", unassigned[0])
        warnings.append("Only one hip marker remaining — one hip missing.")
        print(f" 12. hip_a:           {labels['hip_a']} (only one hip found)")
    else:
        warnings.append("No hip markers remaining.")

    labels["_warnings"] = warnings
    return labels


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from read_serve import read_serve

    if len(sys.argv) < 2:
        print("Usage: python label_markers.py <path_to_csv>")
        sys.exit(1)

    data = read_serve(sys.argv[1])

    print("=== Labeling ===")
    result = label_markers(data)

    print("\n=== Final Assignments ===")
    for part in BODY_PARTS:
        marker = result.get(part)
        if marker:
            tx, ty, tz = _mean_position(data, marker)
            print(f"  {part:<20} → {marker}  (TX={tx:.1f}, TY={ty:.1f}, TZ={tz:.1f})")
        else:
            print(f"  {part:<20} → None  ✗ MISSING")

    if result["_warnings"]:
        print("\n=== Warnings ===")
        for w in result["_warnings"]:
            print(f"  ⚠ {w}")
    else:
        print("\nNo warnings.")
