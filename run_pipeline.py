"""Run the simplified center-out pipeline in stage order."""

import argparse
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
args = parser.parse_args()

for name, script in STAGES:
    print(f"\n=== {name}: {script} ===", flush=True)
    subprocess.run([sys.executable, str(SCRIPT_DIR / script)], cwd=ROOT, check=True)
    if name == args.through:
        break
