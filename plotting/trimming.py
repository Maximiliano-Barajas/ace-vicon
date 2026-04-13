import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow imports from the dtw folder where load_multi_serve lives
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "dtw"))
from load_data import load_multi_serve

# Load one serve
csv_path = os.path.join(
    os.path.dirname(__file__), "markers", "unmarked_edited", "serve4.csv"
)
serve = load_multi_serve(csv_path)

# Pick a marker and extract its coordinates
marker = "right_hand"
TX = serve[marker]["TX"]
TY = serve[marker]["TY"]
TZ = serve[marker]["TZ"]
frames = serve["frames"]

# Calculate speed frame by frame
dX = np.diff(TX)
dY = np.diff(TY)
dZ = np.diff(TZ)

speed = np.sqrt(dX**2 + dY**2 + dZ**2)  # mm per frame

# Smooth the speed signal
smoothed_speed = pd.Series(speed).rolling(window=10, center=True).mean().values

# frames has one more entry than speed (diff shortens by 1); align on mid-frames
speed_frames = frames[:-1]

# Plot
plt.figure(figsize=(12, 5))
plt.plot(
    speed_frames, smoothed_speed, linewidth=1.5, label="smoothed speed (window=10)"
)
plt.xlabel("Frame number")
plt.ylabel("Speed (mm / frame)")
plt.title(f"Hand speed over time  [marker: {marker}]")
plt.legend()
plt.tight_layout()
plt.show()


threshold = np.nanmax(smoothed_speed) * 0.05
above = np.where(smoothed_speed > threshold)[0]
print(f"Threshold: {threshold:.1f} mm/frame")
print(
    f"Frame range above threshold: {speed_frames[above[0]]:.0f} – {speed_frames[above[-1]]:.0f}"
)
print(f"Frames above threshold:\n{speed_frames[above]}")
