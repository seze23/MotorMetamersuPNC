#!/usr/bin/env bash
# Full grid run for the preliminary distortion heatmap: 24 directions x 3
# radii (72 new spatial points), single checkpoint (0_9, already validated).
# Run from dataexp/centerout/ inside the cluster repo, after conda activate.
set -euo pipefail

# Self-activate the conda env regardless of what the calling shell already
# has active -- a tmux new-session does NOT inherit a prior `conda activate`,
# so relying on the caller having done it already is fragile.
if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV:-}" != "proprioception" ]; then
    source /opt/anaconda3-2023.03/etc/profile.d/conda.sh
    conda activate proprioception
fi
python3 -c "import opensim" 2>/dev/null || {
    echo "FATAL: opensim not importable even after activating proprioception env. Aborting.";
    exit 1;
}

echo "=== [0/7] Spindle timing probe (quick, reuses existing extracted data) ==="
python3 spindle_timing_probe.py || echo "(probe failed/skipped, continuing to main run)"

echo "=== [1/7] Generate 24x3 grid targets (72 points) ==="
python3 generate_grid_targets.py

echo "=== [2/7] Analytic IK on grid targets ==="
python3 ikcenterout.py

echo "=== [3/7] Write .mot files ==="
python3 gencenterout.py

echo "=== [4/7] Validate IK plausibility ==="
python3 validate_ik.py || echo "(validation warnings -- review before trusting the heatmap)"

echo "=== [5/7] Extract muscle fiber lengths (OpenSim, dominant cost ~15s/point) ==="
python3 extractcenterout.py

echo "=== [6/7] Compute spindle firing rates ==="
python3 computefrcenterout.py

echo "=== [7/7] CNN inference (checkpoint 0_9 only, ~2min for full grid) ==="
python3 centeroutinference.py

echo "=== Build distortion heatmap figure ==="
python3 heatmap_plot.py

echo "=== DONE. Key outputs: ==="
echo "  distortion_heatmap.png"
echo "  distortion_heatmap_data.csv"
echo "  summary_all_directions.csv (now includes grid points g<deg>_r<radius> alongside original 8)"
