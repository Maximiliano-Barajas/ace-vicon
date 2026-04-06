import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from load_data import load_all_markers

bones = [
    ("head", "chest"),
    ("chest", "left_shoulder"),
    ("chest", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "left_hand"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),
    ("left_hip", "right_hip"),
    ("chest", "left_hip"),
    ("chest", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_foot"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_foot"),
]

markers = load_all_markers("markers/serve1")

# Determine total frames (min length across all joints to stay in bounds)
n_frames = min(len(df) for df in markers.values())

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Set fixed axis limits based on full data range
all_x = np.concatenate([df["tx"].dropna().values for df in markers.values()])
all_y = np.concatenate([df["ty"].dropna().values for df in markers.values()])
all_z = np.concatenate([df["tz"].dropna().values for df in markers.values()])


def padded_limits(data, pad=0.08):
    lo, hi = data.min(), data.max()
    margin = (hi - lo) * pad
    return lo - margin, hi + margin


x_lim = padded_limits(all_x)
y_lim = padded_limits(all_y)
z_lim = padded_limits(all_z)

x_range = x_lim[1] - x_lim[0]
y_range = y_lim[1] - y_lim[0]
z_range = z_lim[1] - z_lim[0]


def get_pos(joint, frame_idx):
    row = markers[joint].iloc[frame_idx]
    return float(row["tx"]), float(row["ty"]), float(row["tz"])


def apply_axes():
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_box_aspect([x_range, y_range, z_range])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


apply_axes()


def update(frame_idx):
    ax.cla()
    apply_axes()
    ax.set_title(f"Frame {frame_idx + 1}")
    # Draw joints
    for joint in markers:
        x, y, z = get_pos(joint, frame_idx)
        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
            ax.scatter(x, y, z, s=20)

    # Draw bones
    for start, end in bones:
        if start not in markers or end not in markers:
            continue
        x0, y0, z0 = get_pos(start, frame_idx)
        x1, y1, z1 = get_pos(end, frame_idx)
        if any(np.isnan(v) for v in [x0, y0, z0, x1, y1, z1]):
            continue
        ax.plot([x0, x1], [y0, y1], [z0, z1], "b-", linewidth=1.5)


ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=33, repeat=True)

plt.show()
