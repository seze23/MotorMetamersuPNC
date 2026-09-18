"""Cluster path configuration for the center-out pipeline.

Counterpart to the local-workspace paths.py — same interface
(REPO_DIR, CENTEROUT_DIR, MODEL_PATH) so generatereachpath.py, ikcenterout.py,
gencenterout.py, extractcenterout.py, computefrcenterout.py, and
centeroutinference.py run unmodified in both environments.
"""

import os
from pathlib import Path
import re

REPO_DIR = str(Path(__file__).resolve().parents[2])
EXPERIMENT_NAME = os.environ.get("MOTOR_META_EXPERIMENT", "center_out")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", EXPERIMENT_NAME):
    raise ValueError(
        "Experiment name must start with a letter or number and contain only "
        "letters, numbers, underscores, and hyphens."
    )
CENTEROUT_DIR = os.path.join(REPO_DIR, "outputs", EXPERIMENT_NAME)

MODEL_PATH = os.path.join(
    REPO_DIR,
    "MoBL-ARMSDynamicUpperLimb-latest",
    "MoBL-ARMS Upper Extremity Model", "Model", "4.1",
    "DefaultMOBL_ARMS_fixed_41.osim",
)

os.makedirs(CENTEROUT_DIR, exist_ok=True)
