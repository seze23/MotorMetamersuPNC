"""Cluster path configuration for the center-out pipeline.

Counterpart to the local-workspace paths.py — same interface
(REPO_DIR, CENTEROUT_DIR, MODEL_PATH) so generatereachpath.py, ikcenterout.py,
gencenterout.py, extractcenterout.py, computefrcenterout.py, and
centeroutinference.py run unmodified in both environments.
"""

import os
from pathlib import Path

REPO_DIR = str(Path(__file__).resolve().parents[2])
CENTEROUT_DIR = os.path.join(REPO_DIR, "outputs")

MODEL_PATH = os.path.join(
    REPO_DIR,
    "MoBL-ARMSDynamicUpperLimb-latest",
    "MoBL-ARMS Upper Extremity Model", "Model", "4.1",
    "DefaultMOBL_ARMS_fixed_41.osim",
)

os.makedirs(CENTEROUT_DIR, exist_ok=True)
