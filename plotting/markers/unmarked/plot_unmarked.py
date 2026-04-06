import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from load_unmarked import load_unmarked_csv

# COLOR KEY (assigned by marker index order in the CSV)
# Marker 1  (Track 445) = darkgreen (head)
# Marker 2  (Track 450) = red (left shoulder)
# Marker 3  (Track 467) = blue (right elbow)
# Marker 4  (Track 496) = orange (left elbow)
# Marker 5  (Track 510) = purple (chest)
# Marker 6  (Track 529) = brown (right shoulder)
# Marker 7  (Track 538) = deeppink (right knee)
# Marker 8  (Track 539) = cyan (right hand)
# Marker 9  (Track 552) = gold (left foot)
# Marker 10 (Track 555) = navy (left knee)
# Marker 11 (Track 561) = lime (right hip)
# Marker 12 (Track 562) = coral (left hand)
# Marker 13 (Track 568) = teal (left hip)

COLORS = [
    "darkgreen",
    "red",
    "blue",
    "orange",
    "purple",
    "brown",
    "deeppink",
    "cyan",
    "gold",
    "navy",
    "lime",
    "coral",
    "teal",
]

csv_path = os.path.join(os.path.dirname(__file__), "serve3.csv")
data = load_unmarked_csv(csv_path)

marker_names = [k for k in data if k != "frames"]
n_frames = len(data["frames"])

# Axis limits from all frames across all markers
all_x = np.concatenate([data[m]["TX"] for m in marker_names])
all_y = np.concatenate([data[m]["TY"] for m in marker_names])
all_z = np.concatenate([data[m]["TZ"] for m in marker_names])


def padded_limits(arr, pad=0.08):
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


x_lim = padded_limits(all_x)
y_lim = padded_limits(all_y)
z_lim = padded_limits(all_z)

x_range = x_lim[1] - x_lim[0]
y_range = y_lim[1] - y_lim[0]
z_range = z_lim[1] - z_lim[0]

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


def apply_axes():
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_box_aspect([x_range, y_range, z_range])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def update(frame_idx):
    ax.cla()
    apply_axes()
    ax.set_title(f'Frame {int(data["frames"][frame_idx])}')

    for i, name in enumerate(marker_names):
        x = data[name]["TX"][frame_idx]
        y = data[name]["TY"][frame_idx]
        z = data[name]["TZ"][frame_idx]
        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            ax.scatter(
                x, y, z, s=40, color=COLORS[i % len(COLORS)], label=f"Marker {i + 1}"
            )

    ax.legend(loc="upper left", fontsize=7)


apply_axes()
ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=33, repeat=True)

plt.tight_layout()
plt.show()
