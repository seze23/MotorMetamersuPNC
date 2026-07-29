# Cluster sync — 2026-07-29

These files bring the CMU Mind cluster copy of
`/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout/` up to date
with the corrected local pipeline: recentered rest posture, analytic-FK inverse
kinematics (0-divergence solver), and the de-noised validator. See
`DRIFT_EXPLAINER.md` (local workspace) for the full scientific rationale.

## Files in this bundle

| File | What changed vs. cluster's current copy |
|---|---|
| `paths.py` | New. Cluster-side path config so all scripts share one source of truth (mirrors the local workspace's `paths.py`). |
| `generatereachpath.py` | Rest posture recentered (elv=45, sh_elv=70, sh_rot=25, elbow=90); imports `paths`. |
| `ikcenterout.py` | Rewritten: solves analytic inverse kinematics directly (was OpenSim marker-tracking IK). See header docstring for rationale. |
| `gencenterout.py` | Unchanged logic, now imports `paths`. |
| `extractcenterout.py` | Unchanged logic, now imports `paths` instead of hardcoded `/home/sydneyez/...`. |
| `computefrcenterout.py` | Unchanged logic, now imports `paths`. |
| `centeroutinference.py` | Unchanged logic, now imports `paths`. |
| `validate_ik.py` | New. Biomechanical plausibility scorecard (EF3D bounds, shoulder_rot degeneracy, reach distance, Z-drift, continuity skipping first 5 IK warm-up frames). |
| `fk_helper.py` | New. Pure-NumPy analytic FK used by stages 1-2 (no h5py/matplotlib dependency). |
| `run_pipeline_cluster.sh` | New. Driver for stages 1-6 in order. |

## What this does NOT touch

- The pretrained checkpoint (`trained_models/.../spatiotemporal_4_8-8-32-64_7171_0_9`) — untouched.
- `utils/`, `inference/`, `train/`, `extract_data/` — untouched (these are imported, not modified).
- Any existing `dataexp/centerout/*` outputs — copying these files does not delete anything. If you want a clean rerun (recommended, since old outputs used the old rest posture/solver), run the "Clean Slate Execution" commands from CLAUDE.md before Step 1.

## Commands (run these yourself after `ssh`)

```bash
# 1. From your LOCAL machine, copy this bundle to the cluster
scp -r ~/Downloads/motormetamers-centerout/cluster_sync/* \
    sydneyez@mind.cs.cmu.edu:/home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout/

# 2. SSH in
ssh sydneyez@mind.cs.cmu.edu
ssh mind-0-18   # or whichever node has the env

# 3. Activate environment
source /opt/anaconda3-2023.03/etc/profile.d/conda.sh
conda activate proprioception
cd /home/sydneyez/sydneyez/ProprioceptiveIllusions/dataexp/centerout

# 4. (Recommended) clean slate before rerun — old outputs used the old rest
#    posture and OpenSim marker-IK solver, and mixing them with the new files
#    would silently produce inconsistent artifacts.
rm -f desired_xyz_*.npz ik_*.npz center_out_*.mot center_out_*_extracted.npz \
      center_out_*_spindles.npz spindles_*.png pred_vs_truth_*.png \
      panel_b_trajectories.png results_*.csv summary_all_directions.csv

# 5. Run everything (long-running — use tmux)
tmux new -s centerout
chmod +x run_pipeline_cluster.sh
./run_pipeline_cluster.sh
# Ctrl+B then D to detach; `tmux attach -t centerout` to reconnect
```

## After it finishes

Report back (or paste) the terminal summary table from `centeroutinference.py`
and/or the contents of `summary_all_directions.csv` — that's the first real
(non-fabricated) RMSE data for this corrected pipeline, and is needed before
any directional-bias interpretation (tasks #10-12, #21-22) can proceed.
