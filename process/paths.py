"""Shared path configuration for the experiment pipeline.

Counterpart to the local-workspace paths.py — same interface
(REPO_DIR, CENTEROUT_DIR, MODEL_PATH) shared by every processing stage.
"""

import os
from pathlib import Path
import re
import yaml

REPO_DIR = str(Path(__file__).resolve().parents[1])
CONFIG_PATH = os.environ.get(
    "MOTOR_META_CONFIG", os.path.join(REPO_DIR, "experiments", "center_out.yaml")
)
with open(CONFIG_PATH, encoding="utf-8") as config_file:
    EXPERIMENT_CONFIG = yaml.safe_load(config_file)
EXPERIMENT_CONFIG["_config_path"] = os.path.abspath(CONFIG_PATH)

EXPERIMENT_NAME = os.environ.get(
    "MOTOR_META_EXPERIMENT", EXPERIMENT_CONFIG.get("experiment", "center_out")
)
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", EXPERIMENT_NAME):
    raise ValueError(
        "Experiment name must start with a letter or number and contain only "
        "letters, numbers, underscores, and hyphens."
    )
CENTEROUT_DIR = os.path.join(REPO_DIR, "outputs", EXPERIMENT_NAME)
PATHS_DIR = os.path.join(CENTEROUT_DIR, "paths")
IK_DIR = os.path.join(CENTEROUT_DIR, "ik")
MOTIONS_DIR = os.path.join(CENTEROUT_DIR, "motions")
MUSCLES_DIR = os.path.join(CENTEROUT_DIR, "muscles")
SPINDLES_DIR = os.path.join(CENTEROUT_DIR, "spindles")
PREDICTIONS_DIR = os.path.join(CENTEROUT_DIR, "predictions")
FIGURES_DIR = os.path.join(CENTEROUT_DIR, "figures")
MANIFEST_PATH = os.path.join(CENTEROUT_DIR, "manifest.yaml")

MODEL_PATH = os.path.join(
    REPO_DIR,
    "MoBL-ARMSDynamicUpperLimb-latest",
    "MoBL-ARMS Upper Extremity Model", "Model", "4.1",
    "DefaultMOBL_ARMS_fixed_41.osim",
)

for directory in (
    CENTEROUT_DIR, PATHS_DIR, IK_DIR, MOTIONS_DIR, MUSCLES_DIR,
    SPINDLES_DIR, PREDICTIONS_DIR, FIGURES_DIR,
):
    os.makedirs(directory, exist_ok=True)
