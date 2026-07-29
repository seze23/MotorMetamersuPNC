#!/usr/bin/env bash
# Full center-out pipeline on the CMU Mind cluster, stages 1-6.
# Run from dataexp/centerout/ inside the cluster repo, after conda activate.
set -euo pipefail

echo "=== Stage 1: generate reach paths (analytic-FK star, recentered rest) ==="
python3 generatereachpath.py

echo "=== Stage 2: analytic inverse kinematics (0-divergence solver) ==="
python3 ikcenterout.py

echo "=== Stage 3: write .mot files ==="
python3 gencenterout.py

echo "=== Validation: biomechanical plausibility ==="
python3 validate_ik.py

echo "=== Stage 4: extract muscle fiber lengths (OpenSim) ==="
python3 extractcenterout.py

echo "=== Stage 5: compute spindle firing rates ==="
python3 computefrcenterout.py

echo "=== Stage 6: CNN inference + figures ==="
python3 centeroutinference.py

echo "=== Done. Outputs in dataexp/centerout/: results_*.csv, summary_all_directions.csv, panel_b_trajectories.png ==="
