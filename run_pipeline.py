"""Run the simplified center-out pipeline in stage order."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT / "process"
STAGES = [
    ("inverse-kinematics", "ikcenterout.py"),
    ("motion", "gencenterout.py"),
    ("muscle-lengths", "extractcenterout.py"),
    ("muscle-signals", "computefrcenterout.py"),
    ("inference", "centeroutinference.py"),
]

parser = argparse.ArgumentParser()
stage_names = ["path"] + [name for name, _ in STAGES]
parser.add_argument("--through", choices=stage_names, default="inference")
parser.add_argument(
    "--config",
    default="experiments/center_out.yaml",
    help="Experiment YAML containing the path generator and its parameters",
)
parser.add_argument(
    "--experiment",
    default=None,
    help="Optional override for the experiment name in the YAML",
)
args = parser.parse_args()

config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
with open(config_path, encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)
experiment_name = args.experiment or config.get("experiment")
if not experiment_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in experiment_name) or not experiment_name[0].isalnum():
    parser.error("--experiment must start with a letter or number and contain only letters, numbers, underscores, and hyphens")

generator = (ROOT / config["path"]["generator"]).resolve()
if not generator.is_file():
    parser.error(f"Configured path generator does not exist: {generator}")

child_env = os.environ.copy()
child_env["MOTOR_META_EXPERIMENT"] = experiment_name
child_env["MOTOR_META_CONFIG"] = str(config_path)
print(f"Experiment outputs: {ROOT / 'outputs' / experiment_name}", flush=True)

all_stages = [("path", str(generator))] + [
    (name, str(SCRIPT_DIR / script)) for name, script in STAGES
]
for name, script in all_stages:
    print(f"\n=== {name}: {script} ===", flush=True)
    subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        env=child_env,
        check=True,
    )
    if name == args.through:
        break
