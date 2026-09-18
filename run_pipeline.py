"""Run the simplified center-out pipeline in stage order."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT / "dataexp" / "centerout"
STAGES = [
    ("path", "generatereachpath.py"),
    ("inverse-kinematics", "ikcenterout.py"),
    ("motion", "gencenterout.py"),
    ("muscle-lengths", "extractcenterout.py"),
    ("muscle-signals", "computefrcenterout.py"),
    ("inference", "centeroutinference.py"),
]

parser = argparse.ArgumentParser()
parser.add_argument("--through", choices=[name for name, _ in STAGES], default="inference")
parser.add_argument(
    "--experiment",
    default="center_out",
    help="Output folder name beneath outputs/ (default: center_out)",
)
args = parser.parse_args()

if not args.experiment or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.experiment) or not args.experiment[0].isalnum():
    parser.error("--experiment must start with a letter or number and contain only letters, numbers, underscores, and hyphens")

child_env = os.environ.copy()
child_env["MOTOR_META_EXPERIMENT"] = args.experiment
print(f"Experiment outputs: {ROOT / 'outputs' / args.experiment}", flush=True)

for name, script in STAGES:
    print(f"\n=== {name}: {script} ===", flush=True)
    subprocess.run(
        [sys.executable, str(SCRIPT_DIR / script)],
        cwd=ROOT,
        env=child_env,
        check=True,
    )
    if name == args.through:
        break
