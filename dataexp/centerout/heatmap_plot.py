"""
Build the directional-distortion heatmap from the 24-direction x 3-radius
grid run. Reads results_g<deg>_r<radius>.csv (produced by
centeroutinference.py's main per-direction loop, unmodified -- it discovers
these via glob alongside the original 8 named directions).

For each grid point, takes the L2 error at the peak-reach frame (index 515 =
N_HOLD_PRE + N_REACH - 1, matching the reach-only isolation used for the
original 8-direction figure) and the true reach-relative (x,y) position, then
builds:
  1. A scatter plot colored by L2 (raw data, no interpolation assumptions).
  2. A griddata-interpolated smooth heatmap over the same points.

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout
  python3 heatmap_plot.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from paths import CENTEROUT_DIR

N_HOLD_PRE = 396
N_REACH = 120
PEAK_FRAME = N_HOLD_PRE + N_REACH - 1  # 515, matches reach-only isolation

csv_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "results_g*.csv")))
if not csv_files:
    raise FileNotFoundError(
        f"No results_g*.csv in {CENTEROUT_DIR} run generate_grid_targets.py "
        "through centeroutinference.py first."
    )
print(f"Found {len(csv_files)} grid-point result files.")

rows = []
for f in csv_files:
    df = pd.read_csv(f)
    onset = df.iloc[N_HOLD_PRE]
    peak = df.iloc[PEAK_FRAME]
    x_rel = peak["true_wrist_X_cm"] - onset["true_wrist_X_cm"]
    y_rel = peak["true_wrist_Y_cm"] - onset["true_wrist_Y_cm"]
    l2 = peak["l2_distance_cm"]
    name = os.path.basename(f).replace("results_", "").replace(".csv", "")
    rows.append(dict(name=name, x=x_rel, y=y_rel, l2=l2))

grid_df = pd.DataFrame(rows)
out_csv = os.path.join(CENTEROUT_DIR, "distortion_heatmap_data.csv")
grid_df.to_csv(out_csv, index=False)
print(f"Saved {out_csv}")
print(grid_df.describe())

x = grid_df["x"].values
y = grid_df["y"].values
l2 = grid_df["l2"].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sc = ax1.scatter(x, y, c=l2, cmap="inferno", s=80, edgecolors="black", linewidth=0.4)
ax1.set_title("Distortion heatmap (raw grid points)")
ax1.set_xlabel("X (cm)")
ax1.set_ylabel("Y (cm)")
ax1.set_aspect("equal")
ax1.axhline(0, c='gray', linewidth=0.5, alpha=0.5)
ax1.axvline(0, c='gray', linewidth=0.5, alpha=0.5)
plt.colorbar(sc, ax=ax1, label="Peak-reach L2 error (cm)")

grid_x, grid_y = np.mgrid[x.min():x.max():200j, y.min():y.max():200j]
grid_z = griddata((x, y), l2, (grid_x, grid_y), method="cubic")
im = ax2.pcolormesh(grid_x, grid_y, grid_z, cmap="inferno", shading="auto")
ax2.scatter(x, y, c="white", s=6, alpha=0.6)
ax2.set_title("Distortion heatmap (interpolated)")
ax2.set_xlabel("X (cm)")
ax2.set_ylabel("Y (cm)")
ax2.set_aspect("equal")
plt.colorbar(im, ax=ax2, label="Peak-reach L2 error (cm), interpolated")

fig.suptitle("Center-out reconstruction distortion field (preliminary, seed 0_9 only, single run)")
plt.tight_layout()
out_png = os.path.join(CENTEROUT_DIR, "distortion_heatmap.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_png}")
