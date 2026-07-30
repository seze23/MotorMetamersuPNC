"""
Poster figures, aesthetics-only -- reads multi_seed_results.csv (produced by
multi_seed_inference.py) and produces the polished panels for the poster.
No inference logic here; pure plotting.

Exploratory pass, per researcher: if these look noisy/confusing, fall back
to the existing single-checkpoint (0_9) figures already validated
(distortion_heatmap.png, panel_b_reach_only_trajectories.png).

Produces:
  1. poster_heatmap_leftpanel.png -- mean L2 (at peak-reach frame) across
     all 20 seeds, per grid point, scatter colored by magnitude (matches
     the "raw grid points" left sub-panel style of the original heatmap,
     no interpolation).
  2. poster_arrow_plot.png -- Fig 3a-style error vectors. Per grid point:
     20 individual-seed arrows in light gray + 1 bold black arrow for the
     across-seed mean. Tail = predicted location, head = true target
     location (confirmed convention 2026-07-30 -- reversed from Wang et
     al.'s own tail=true/head=predicted, chosen deliberately so the
     visual direction of our arrows in the upper-right/far region matches
     theirs for easy side-by-side comparison; caption this as an "error
     vector," not literally their "bias vector").

Run:
  cd /home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout
  python3 poster_figures.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CENTEROUT_DIR = "/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout"
RESULTS_CSV = os.path.join(CENTEROUT_DIR, "multi_seed_results.csv")

df_all = pd.read_csv(RESULTS_CSV)
print(f"Loaded {len(df_all)} rows, {df_all['seed'].nunique()} seeds, {df_all['direction'].nunique()} grid points")

# Found 2026-07-30: spindle data was generated with one fixed COEF_SEED (0).
# Checkpoints trained on a different COEF_SEED are fed input from a
# mismatched afferent-sampling distribution -- confirmed directly (e.g.
# seed 3_1's predicted X values are negative when every true X is
# positive). Only the 4 COEF_SEED=0 checkpoints are a valid comparison.
VALID_COEF_SEED = 0
df = df_all[df_all["coef_seed"] == VALID_COEF_SEED].copy()
print(f"Filtered to COEF_SEED={VALID_COEF_SEED}: {len(df)} rows, seeds = {sorted(df['seed'].unique())}")

# Center each seed's true/pred positions relative to that seed's own true
# center-hold position isn't available here directly (peak-reach frame
# only) -- true_x/true_y across seeds for the SAME direction should be
# ~identical (analytic-FK labels don't depend on the seed), so use one
# seed's true position as the reference center via the rest-hold value
# already baked into the grid design (all directions share the same
# analytic-FK center). We center on the mean true position across all
# grid points x seeds' true values at direction "0_right"-like baseline
# is not available here -- instead, plot RAW peak-reach coordinates
# directly (already centered in the analytic-FK world frame from
# generatereachpath.py), no additional re-centering needed.

# ---- Figure 1: heatmap, left-panel style, mean L2 across seeds ----
mean_by_point = df.groupby("direction").agg(
    true_x=("true_x", "mean"), true_y=("true_y", "mean"),
    mean_l2=("l2_at_peak", "mean"), std_l2=("l2_at_peak", "std"),
).reset_index()

fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(mean_by_point["true_x"], mean_by_point["true_y"],
                 c=mean_by_point["mean_l2"], cmap="inferno", s=120,
                 edgecolors="black", linewidth=0.6)
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_title(f"Peak-reach L2 error, mean across {df['seed'].nunique()} seeds\n(raw grid points)")
ax.set_aspect("equal")
ax.axhline(0, c="gray", linewidth=0.5, alpha=0.4)
ax.axvline(0, c="gray", linewidth=0.5, alpha=0.4)
plt.colorbar(sc, ax=ax, label="Mean L2 error (cm)")
plt.tight_layout()
out1 = os.path.join(CENTEROUT_DIR, "poster_heatmap_leftpanel.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out1}")

# ---- Figure 2: arrow plot, Fig 3a-style ----
fig, ax = plt.subplots(figsize=(9, 9))

for direction, group in df.groupby("direction"):
    true_x = group["true_x"].mean()
    true_y = group["true_y"].mean()

    # individual-seed arrows, light gray
    for _, row in group.iterrows():
        dx = true_x - row["pred_x"]
        dy = true_y - row["pred_y"]
        ax.annotate("", xy=(true_x, true_y), xytext=(row["pred_x"], row["pred_y"]),
                    arrowprops=dict(arrowstyle="-|>", color="lightgray",
                                     lw=0.8, alpha=0.6, mutation_scale=8))

    # mean arrow, bold black -- tail = mean predicted, head = true target
    mean_pred_x = group["pred_x"].mean()
    mean_pred_y = group["pred_y"].mean()
    ax.annotate("", xy=(true_x, true_y), xytext=(mean_pred_x, mean_pred_y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2.0,
                                 mutation_scale=16))
    ax.scatter([true_x], [true_y], c="black", s=15, zorder=5)

ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_title(f"Error vectors: predicted -> true target, peak-reach frame\n"
             f"gray = {df['seed'].nunique()} individual seeds, black = mean")
ax.set_aspect("equal")
ax.axhline(0, c="gray", linewidth=0.5, alpha=0.4)
ax.axvline(0, c="gray", linewidth=0.5, alpha=0.4)
plt.tight_layout()
out2 = os.path.join(CENTEROUT_DIR, "poster_arrow_plot.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out2}")

# ---- Seed ranking printout ----
# CAVEAT: confounded across COEF_SEED (see filter above) -- non-zero
# COEF_SEED checkpoints score badly because of the input mismatch, not
# network quality. Only the within-COEF_SEED=0 comparison is valid; do not
# cite the full 20-seed ranking as evidence 0_9 is the best-trained network.
ranking_csv = os.path.join(CENTEROUT_DIR, "multi_seed_ranking.csv")
if os.path.exists(ranking_csv):
    ranking = pd.read_csv(ranking_csv)
    print("\nFull ranking (confounded across COEF_SEED, see caveat above):")
    print(ranking[["rank", "seed", "overall_mean_l2"]].to_string(index=False))
    print(f"\nValid comparison, within COEF_SEED={VALID_COEF_SEED} only:")
    valid_ranking = ranking[ranking["coef_seed"] == VALID_COEF_SEED].sort_values("overall_mean_l2").reset_index(drop=True)
    print(valid_ranking[["seed", "overall_mean_l2"]].to_string(index=False))
