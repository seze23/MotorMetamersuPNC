"""
Diagnostic: does reconstruction error spike specifically at the temporal
boundaries of the 1152-frame input sequence (frames near 0 and near 1151),
consistent with a zero-padding artifact in the spatiotemporal CNN's Conv1d
layers -- as opposed to simply being worse during active movement?

Method: for every already-run trial (original 8 directions + 72-point grid),
compare mean L2 error in a window right at each temporal edge against a
window of the same size, same behavioral state (static hold, not reach),
but away from the edge:

  edge_start   = frames [0:10]      (right at the start of the sequence)
  interior_pre = frames [200:210]   (still pre-reach hold, but far from edge)
  edge_end     = frames [1142:1152] (right at the end of the sequence)
  interior_post= frames [900:910]   (still post-reach hold, but far from edge)

All four windows are within HOLD phases (no reach movement), so a genuine
edge-vs-interior difference isolates the padding-boundary effect from the
already-known reach-phase distortion, rather than conflating the two.

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout
  python3 boundary_artifact_check.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from paths import CENTEROUT_DIR

csv_files = sorted(glob.glob(os.path.join(CENTEROUT_DIR, "results_*.csv")))
if not csv_files:
    raise FileNotFoundError(f"No results_*.csv in {CENTEROUT_DIR}")
print(f"Found {len(csv_files)} trial result files.")

rows = []
all_l2_by_frame = []
for f in csv_files:
    df = pd.read_csv(f)
    l2 = df["l2_distance_cm"].values
    if len(l2) != 1152:
        print(f"  SKIP {os.path.basename(f)}: unexpected length {len(l2)}")
        continue
    name = os.path.basename(f).replace("results_", "").replace(".csv", "")
    rows.append(dict(
        name=name,
        edge_start=l2[0:10].mean(),
        interior_pre=l2[200:210].mean(),
        edge_end=l2[1142:1152].mean(),
        interior_post=l2[900:910].mean(),
    ))
    all_l2_by_frame.append(l2)

summary = pd.DataFrame(rows)
out_csv = os.path.join(CENTEROUT_DIR, "boundary_artifact_check.csv")
summary.to_csv(out_csv, index=False)
print(f"Saved {out_csv}\n")

print("=" * 60)
print("EDGE vs INTERIOR (both within static-hold phases, no movement)")
print("=" * 60)
es, ip = summary["edge_start"], summary["interior_pre"]
ee, ipo = summary["edge_end"], summary["interior_post"]
print(f"Start-of-sequence: edge={es.mean():.4f}cm  interior={ip.mean():.4f}cm  "
      f"diff={ (es-ip).mean():+.4f}cm  ({(es > ip).sum()}/{len(es)} trials edge > interior)")
print(f"End-of-sequence:   edge={ee.mean():.4f}cm  interior={ipo.mean():.4f}cm  "
      f"diff={ (ee-ipo).mean():+.4f}cm  ({(ee > ipo).sum()}/{len(ee)} trials edge > interior)")

# Full-sequence mean L2 vs frame index, averaged across all trials -- the
# clearest visual signature of a padding artifact would be a sharp spike
# right at frame 0 and/or frame 1151 that doesn't match the reach-phase shape.
all_l2_by_frame = np.array(all_l2_by_frame)  # (n_trials, 1152)
mean_l2 = all_l2_by_frame.mean(axis=0)
std_l2 = all_l2_by_frame.std(axis=0)

fig, ax = plt.subplots(figsize=(12, 5))
t = np.arange(1152)
ax.plot(t, mean_l2, c="crimson", linewidth=1.2)
ax.fill_between(t, mean_l2 - std_l2, mean_l2 + std_l2, color="crimson", alpha=0.15)
for edge in [0, 1151]:
    ax.axvline(edge, c="black", linestyle="--", linewidth=0.8, alpha=0.6)
for phase_edge in [396, 516, 636, 756]:
    ax.axvline(phase_edge, c="gray", linestyle=":", linewidth=0.6, alpha=0.4)
ax.set_xlabel("Frame index (0-1151)")
ax.set_ylabel("Mean L2 error across all trials (cm)")
ax.set_title(f"L2 error vs. frame index, averaged across {len(all_l2_by_frame)} trials\n"
             "(dashed = sequence boundaries; dotted = reach/hold/return phase edges)")
plt.tight_layout()
out_png = os.path.join(CENTEROUT_DIR, "boundary_artifact_check.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved {out_png}")
