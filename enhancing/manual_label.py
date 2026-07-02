"""
manual_label.py

Interactive 3D tool for manually labeling unlabeled marker CSVs.

- Plots all markers in 3D
- Play/pause animation with a frame slider to scrub through frames
- ←/→ arrow keys to cycle through markers
- Type 1-14 to assign that body part to the selected marker
- Space to play/pause, Escape to clear selection
- Drag mouse to rotate the 3D view
- Saves a labeled CSV

Usage:
    python manual_label.py <unlabeled_csv>

Example:
    python manual_label.py data\\14unL.csv

Requirements:
    pip install matplotlib numpy pandas
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from read_serve import read_serve

BODY_PARTS = [
    "head",           # 1
    "chest",          # 2
    "leftshoulder",   # 3
    "rightshoulder",  # 4
    "leftelbow",      # 5
    "rightelbow",     # 6
    "lefthand",       # 7
    "righthand",      # 8
    "lefthip",        # 9
    "righthip",       # 10
    "leftknee",       # 11
    "rightknee",      # 12
    "leftfoot",       # 13
    "rightfoot",      # 14
]

MARKER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#f7b733", "#00c957", "#6a0dad", "#e6194b",
]

LABELED_COLOR  = "#00ff00"
SELECTED_COLOR = "#ffffff"

PLAYBACK_FPS         = 30
PLAYBACK_INTERVAL_MS = max(1, int(1000 / PLAYBACK_FPS))


def get_stable_frame(data: dict, marker_keys: list) -> int:
    n_frames = len(data["frames"])
    best_frame, best_count = 0, 0
    for fi in range(n_frames):
        count = sum(1 for m in marker_keys if not np.isnan(data[m]["TX"][fi]))
        if count > best_count:
            best_count, best_frame = count, fi
        if best_count == len(marker_keys):
            break
    return best_frame


def get_position(data: dict, marker: str, frame: int) -> tuple:
    return (data[marker]["TX"][frame],
            data[marker]["TY"][frame],
            data[marker]["TZ"][frame])


def save_labeled_csv(csv_path: str, labels: dict, output_path: str):
    data = read_serve(csv_path)
    frames = data["frames"]
    n_frames = len(frames)
    marker_keys = [k for k in data if k.startswith("marker_")]
    marker_to_part = {k: v for k, v in labels.items() if v is not None}

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
            tx, ty, tz = data[m]["TX"][fi], data[m]["TY"][fi], data[m]["TZ"][fi]
            row.append("" if np.isnan(tx) else f"{tx:.3f}")
            row.append("" if np.isnan(ty) else f"{ty:.3f}")
            row.append("" if np.isnan(tz) else f"{tz:.3f}")
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False, header=False)
    print(f"Saved labeled CSV: {output_path}")


def run_labeler(csv_path: str):
    data = read_serve(csv_path)
    marker_keys = sorted([k for k in data if k.startswith("marker_")])
    n_markers   = len(marker_keys)
    n_frames    = len(data["frames"])

    # Precompute all positions for fast scrubbing
    all_positions = []
    for fi in range(n_frames):
        pos = {m: get_position(data, m, fi) for m in marker_keys}
        all_positions.append(pos)

    start_frame = get_stable_frame(data, marker_keys)

    print(f"Loaded {n_frames} frames, {n_markers} markers.")
    print(f"Starting on frame {int(data['frames'][start_frame])} ({start_frame+1}/{n_frames})")
    print()
    print("Instructions:")
    print("  ←/→  arrow keys : cycle through markers")
    print("  1-14        : assign body part to selected marker")
    print("  Space       : play / pause animation")
    print("  Esc         : clear selection")
    print("  Mouse drag  : rotate the 3D view")
    print("  Save button : export labeled CSV")
    print()

    # ── STATE ──────────────────────────────────────────────────────────────
    state = {
        "frame_idx":      start_frame,
        "playing":        False,
        "selected":       None,
        "labels":         {},          # marker_key -> body part name
        "part_to_marker": {},          # body part  -> marker_key
        "key_buf":        "",          # for two-digit entry (10-14)
    }

    # ── FIGURE LAYOUT ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor("#1a1a2e")

    # 3D plot — room at bottom for playback controls
    ax = fig.add_axes([0.0, 0.18, 0.65, 0.78], projection="3d")
    ax.set_facecolor("#1a1a2e")
    ax.view_init(elev=15, azim=-60)

    # ── PLAYBACK CONTROLS ──────────────────────────────────────────────────
    ax_playpause = fig.add_axes([0.02, 0.06, 0.10, 0.07])
    btn_playpause = widgets.Button(ax_playpause, "▶  Play",
                                   color="#0f3460", hovercolor="#533483")
    btn_playpause.label.set_color("white")
    btn_playpause.label.set_fontsize(10)

    ax_slider = fig.add_axes([0.15, 0.07, 0.48, 0.04])
    ax_slider.set_facecolor("#16213e")
    frame_slider = widgets.Slider(
        ax_slider, label="",
        valmin=0, valmax=n_frames - 1,
        valinit=start_frame, valstep=1,
        color="#533483",
    )
    frame_slider.valtext.set_color("white")
    frame_slider.label.set_color("white")

    ax_framectr = fig.add_axes([0.64, 0.06, 0.01, 0.05])
    ax_framectr.axis("off")
    frame_counter_text = ax_framectr.text(
        0.0, 0.5, f"1 / {n_frames}",
        color="white", fontsize=8,
        va="center", ha="left",
        transform=ax_framectr.transAxes,
    )

    # ── RIGHT PANEL: numbered body-part list ───────────────────────────────
    ax_list = fig.add_axes([0.68, 0.32, 0.28, 0.60])
    ax_list.set_facecolor("#16213e")
    ax_list.axis("off")
    ax_list.set_title("Body Parts  (type number to assign)",
                      color="white", fontsize=9, pad=6)

    part_texts = []
    for i, part in enumerate(BODY_PARTS):
        t = ax_list.text(
            0.05,
            0.97 - i * (0.97 / len(BODY_PARTS)),
            f"  {i+1:>2}.  {part}",
            color="#aaaacc", fontsize=9, fontfamily="monospace",
            transform=ax_list.transAxes, va="top",
        )
        part_texts.append(t)

    # Clear button
    ax_clear = fig.add_axes([0.68, 0.21, 0.28, 0.08])
    btn_clear = widgets.Button(ax_clear, "Clear Selection  [Esc]",
                               color="#0f3460", hovercolor="#e94560")
    btn_clear.label.set_color("white")

    # Save button
    ax_save = fig.add_axes([0.68, 0.11, 0.28, 0.08])
    btn_save = widgets.Button(ax_save, "Save Labeled CSV",
                              color="#0f3460", hovercolor="#00c957")
    btn_save.label.set_color("white")

    # Status text
    ax_status = fig.add_axes([0.68, 0.01, 0.28, 0.08])
    ax_status.set_facecolor("#16213e")
    ax_status.axis("off")
    status_text = ax_status.text(
        0.5, 0.5, "Press ←/→ to select a marker",
        ha="center", va="center", color="white",
        fontsize=9, wrap=True, transform=ax_status.transAxes,
    )

    # ── HELPERS ────────────────────────────────────────────────────────────
    def update_status(msg, color="white"):
        status_text.set_text(msg)
        status_text.set_color(color)
        fig.canvas.draw_idle()

    def refresh_part_list():
        assigned = set(state["labels"].values())
        for i, t in enumerate(part_texts):
            part = BODY_PARTS[i]
            if part in assigned:
                t.set_color(LABELED_COLOR)
                t.set_text(f"\u2713 {i+1:>2}.  {part}")
            else:
                t.set_color("#aaaacc")
                t.set_text(f"  {i+1:>2}.  {part}")
        fig.canvas.draw_idle()

    def update_frame_counter():
        fi = state["frame_idx"]
        frame_counter_text.set_text(f"{fi + 1} / {n_frames}")

    # ── STABLE GLOBAL BOUNDS ───────────────────────────────────────────────
    _gx, _gy, _gz = [], [], []
    for pos in all_positions:
        for m in marker_keys:
            x, y, z = pos[m]
            if not np.isnan(x):
                _gx.append(x); _gy.append(y); _gz.append(z)

    PAD = 150
    if _gx:
        cx, cy, cz = np.mean(_gx), np.mean(_gy), np.mean(_gz)
        half_span = max(
            max(_gx) - min(_gx),
            max(_gy) - min(_gy),
            max(_gz) - min(_gz),
        ) / 2 + PAD
        GLOBAL_XLIM = (cx - half_span, cx + half_span)
        GLOBAL_YLIM = (cy - half_span, cy + half_span)
        GLOBAL_ZLIM = (cz - half_span, cz + half_span)
    else:
        half_span = 1000
        GLOBAL_XLIM = GLOBAL_YLIM = GLOBAL_ZLIM = (-1000, 1000)

    scatter_objects = {}

    # ── REDRAW ─────────────────────────────────────────────────────────────
    def redraw():
        fi        = state["frame_idx"]
        positions = all_positions[fi]

        ax.cla()
        ax.set_facecolor("#1a1a2e")
        ax.set_xlabel("X (floor)", color="white", labelpad=2)
        ax.set_ylabel("Y (floor)", color="white", labelpad=2)
        ax.set_zlabel("Z (height)", color="white", labelpad=2)
        ax.tick_params(colors="white", labelsize=7, pad=1)
        ax.set_title(
            f"Frame {int(data['frames'][fi])}  ({fi+1}/{n_frames})  —  "
            f"{len(state['labels'])}/{n_markers} labeled",
            color="white", fontsize=11,
        )

        ax.set_xlim(*GLOBAL_XLIM)
        ax.set_ylim(*GLOBAL_YLIM)
        ax.set_zlim(*GLOBAL_ZLIM)
        ax.set_box_aspect((1, 1, 1))

        ax.xaxis.pane.set_facecolor("#1e1e3a")
        ax.yaxis.pane.set_facecolor("#1e1e3a")
        ax.zaxis.pane.set_facecolor("#252540")
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.set_edgecolor("#333355")
            pane.fill = True

        scatter_objects.clear()
        for i, m in enumerate(marker_keys):
            x, y, z = positions[m]
            if np.isnan(x):
                continue

            if m == state["selected"]:
                color, size = SELECTED_COLOR, 200
            elif m in state["labels"]:
                color, size = LABELED_COLOR, 120
            else:
                color, size = MARKER_COLORS[i % len(MARKER_COLORS)], 120

            sc = ax.scatter(x, y, z, s=size, c=color, zorder=5, depthshade=False)
            scatter_objects[m] = sc

            part = state["labels"].get(m, m)
            label_color = SELECTED_COLOR if m == state["selected"] else color
            ax.text(
                x, y, z + half_span * 0.04,
                f" {part}",
                fontsize=8, fontweight="bold", color=label_color,
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="#1a1a2e", edgecolor="none", alpha=0.6),
            )

        update_frame_counter()
        fig.canvas.draw_idle()

    # ── ANIMATION ──────────────────────────────────────────────────────────
    def animation_step(_):
        if not state["playing"]:
            return
        next_fi = (state["frame_idx"] + 1) % n_frames
        state["frame_idx"] = next_fi
        frame_slider.eventson = False
        frame_slider.set_val(next_fi)
        frame_slider.eventson = True
        redraw()

    anim = animation.FuncAnimation(
        fig, animation_step,
        interval=PLAYBACK_INTERVAL_MS,
        cache_frame_data=False, blit=False,
    )

    # ── PLAY/PAUSE ─────────────────────────────────────────────────────────
    def on_playpause(event):
        state["playing"] = not state["playing"]
        if state["playing"]:
            btn_playpause.label.set_text("\u23f8  Pause")
            update_status("Playing\u2026", "#17becf")
        else:
            btn_playpause.label.set_text("\u25b6  Play")
            update_status("Paused.", "white")
        fig.canvas.draw_idle()

    btn_playpause.on_clicked(on_playpause)

    # ── FRAME SLIDER ───────────────────────────────────────────────────────
    def on_slider_change(val):
        fi = int(round(val))
        if fi != state["frame_idx"]:
            state["frame_idx"] = fi
            redraw()

    frame_slider.on_changed(on_slider_change)

    # ── CLICK: select nearest marker ───────────────────────────────────────
    # ── MOUSE: drag anywhere on the 3D axes to rotate (azimuth only) ────────
    FIXED_ELEV   = 15
    rotate_state = {"active": False, "x0": None, "azim0": None}

    def on_mouse_press(event):
        if event.inaxes == ax and event.button == 1:
            rotate_state["active"] = True
            rotate_state["x0"]     = event.x
            rotate_state["azim0"]  = ax.azim

    def on_mouse_move(event):
        if not rotate_state["active"] or event.x is None:
            return
        dx = event.x - rotate_state["x0"]
        ax.view_init(elev=FIXED_ELEV, azim=rotate_state["azim0"] - dx * 0.5)
        fig.canvas.draw_idle()

    def on_mouse_release(event):
        rotate_state["active"] = False

    ax.mouse_init(rotate_btn=None, zoom_btn=None)
    fig.canvas.mpl_connect("button_press_event",   on_mouse_press)
    fig.canvas.mpl_connect("motion_notify_event",  on_mouse_move)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)

    # ── KEYBOARD CONTROLS ──────────────────────────────────────────────────
    # Left/Right arrows  → cycle through markers
    # Space              → play / pause
    # 1-14               → assign body part to selected marker
    # Esc                → clear selection

    def select_marker(marker_key):
        """Highlight a marker and show its current label."""
        state["selected"] = marker_key
        part = state["labels"].get(marker_key, "unassigned")
        idx  = marker_keys.index(marker_key)
        update_status(
            f"[{idx+1}/{n_markers}] {marker_key}\nCurrent: {part}\nType 1-14 to assign",
            "#17becf",
        )
        redraw()

    def do_assign(part_index):
        """Assign BODY_PARTS[part_index] to the currently selected marker."""
        if state["selected"] is None:
            update_status("No marker selected!\nPress ←/→ to pick one.", "#e94560")
            return
        chosen_part = BODY_PARTS[part_index]
        # Remove any existing claim on this part
        if chosen_part in state["part_to_marker"]:
            old_marker = state["part_to_marker"][chosen_part]
            if old_marker != state["selected"]:
                del state["labels"][old_marker]
        # Remove old part for this marker
        old_part = state["labels"].get(state["selected"])
        if old_part and old_part in state["part_to_marker"]:
            del state["part_to_marker"][old_part]
        state["labels"][state["selected"]] = chosen_part
        state["part_to_marker"][chosen_part] = state["selected"]
        assigned_marker = state["selected"]
        update_status(
            f"\u2713 Assigned:\n{assigned_marker} \u2192 {chosen_part}",
            "#00c957",
        )
        # Auto-advance to next unlabeled marker if any remain
        unlabeled = [m for m in marker_keys if m not in state["labels"]]
        if unlabeled:
            state["selected"] = unlabeled[0]
            part = state["labels"].get(unlabeled[0], "unassigned")
            idx  = marker_keys.index(unlabeled[0])
            update_status(
                f"\u2713 {assigned_marker} \u2192 {chosen_part}\n"
                f"Next [{idx+1}/{n_markers}]: {unlabeled[0]}\nType 1-14 to assign",
                "#00c957",
            )
        else:
            state["selected"] = None
            update_status("All markers labeled!\nClick Save to export.", "#00c957")
        refresh_part_list()
        redraw()

    def on_key(event):
        key = event.key

        # ── navigation ────────────────────────────────────────────────────
        if key in ("right", "left"):
            valid = [m for m in marker_keys
                     if not np.isnan(all_positions[state["frame_idx"]][m][0])]
            if not valid:
                return
            if state["selected"] is None or state["selected"] not in valid:
                idx = 0 if key == "right" else len(valid) - 1
            else:
                cur = valid.index(state["selected"])
                idx = (cur + (1 if key == "right" else -1)) % len(valid)
            select_marker(valid[idx])
            return

        # ── play/pause ────────────────────────────────────────────────────
        if key == " ":
            state["playing"] = not state["playing"]
            if state["playing"]:
                btn_playpause.label.set_text("\u23f8  Pause")
                update_status("Playing\u2026", "#17becf")
            else:
                btn_playpause.label.set_text("\u25b6  Play")
                update_status("Paused.", "white")
            fig.canvas.draw_idle()
            return

        # ── clear selection ───────────────────────────────────────────────
        if key == "escape":
            state["selected"] = None
            state["key_buf"]  = ""
            update_status("Selection cleared.\nPress \u2190/\u2192 to select a marker.", "white")
            redraw()
            return

        # ── number assignment (1-14, two-digit buffered) ──────────────────
        if key not in [str(d) for d in range(10)]:
            # Any non-digit flushes a buffered "1" as assignment
            if state["key_buf"] == "1":
                state["key_buf"] = ""
                do_assign(0)
            return

        buf = state["key_buf"] + key
        if len(buf) == 2:
            num = int(buf)
            if 1 <= num <= len(BODY_PARTS):
                state["key_buf"] = ""
                do_assign(num - 1)
                return
            # Not a valid pair — use first digit alone, buffer second
            first = int(buf[0])
            state["key_buf"] = key
            if 1 <= first <= len(BODY_PARTS):
                do_assign(first - 1)
        else:
            num = int(key)
            if num == 0:
                # flush buffered "1" if present
                if state["key_buf"] == "1":
                    state["key_buf"] = ""
                    do_assign(0)   # assign "head" (part 1)
                return
            if num >= 2:
                state["key_buf"] = ""
                do_assign(num - 1)
            else:   # "1" — wait for possible second digit
                state["key_buf"] = key
                update_status(
                    "1 pressed\u2014type 0-4 for 10-14,\nor \u2190/\u2192/other to assign head.",
                    "#17becf",
                )

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ── CLEAR BUTTON ───────────────────────────────────────────────────────
    def on_clear(event):
        state["selected"] = None
        state["key_buf"]  = ""
        update_status("Selection cleared.", "white")
        redraw()

    btn_clear.on_clicked(on_clear)

    # ── SAVE BUTTON ────────────────────────────────────────────────────────
    def on_save(event):
        state["playing"] = False
        btn_playpause.label.set_text("\u25b6  Play")

        n_labeled = len(state["labels"])
        if n_labeled < n_markers:
            update_status(
                f"Only {n_labeled}/{n_markers} labeled.\nLabel all markers first.",
                "#e94560",
            )
            return

        print("\n=== Label Assignments ===")
        for m in marker_keys:
            print(f"  {m:<12} \u2192 {state['labels'].get(m, 'UNASSIGNED')}")

        confirm = input("\nSave labeled CSV? (y/n): ").strip().lower()
        if confirm == "y":
            base     = os.path.splitext(os.path.basename(csv_path))[0]
            out_dir  = os.path.dirname(csv_path)
            out_path = os.path.join(out_dir, f"{base}_labeled.csv")
            save_labeled_csv(csv_path, state["labels"], out_path)
            update_status(f"Saved!\n{os.path.basename(out_path)}", "#00c957")
        else:
            update_status("Save cancelled.", "#e94560")

    btn_save.on_clicked(on_save)

    # ── INITIAL DRAW ───────────────────────────────────────────────────────
    redraw()
    refresh_part_list()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python manual_label.py <unlabeled_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)

    run_labeler(csv_path)
