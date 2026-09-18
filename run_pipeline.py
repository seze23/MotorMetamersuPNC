"""Run the simplified center-out pipeline in stage order.

The :class:`Pipeline` API is also used by ``run_pipeline.ipynb`` so the command
line and notebook interfaces execute exactly the same scripts.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT / "process"
DISPLAY_SCRIPT = SCRIPT_DIR / "utils" / "display_sim.py"
STAGES = [
    ("path", None),
    ("inverse-kinematics", "ikcenterout.py"),
    ("motion", "gencenterout.py"),
    ("muscle-lengths", "extractcenterout.py"),
    ("muscle-signals", "computefrcenterout.py"),
    ("inference", "centeroutinference.py"),
]
STAGE_NAMES = [name for name, _ in STAGES]


def _resolve_config(config_path):
    path = Path(config_path)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


class Pipeline:
    """Execute pipeline stages in child processes using one experiment YAML."""

    def __init__(self, config="experiments/center_out.yaml", experiment=None):
        self.config_path = _resolve_config(config)
        with self.config_path.open(encoding="utf-8") as config_file:
            self.config = yaml.safe_load(config_file)
        self.experiment = experiment or self.config.get("experiment")
        if not self.experiment or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]*", self.experiment
        ):
            raise ValueError(
                "Experiment name must start with a letter or number and contain "
                "only letters, numbers, underscores, and hyphens."
            )
        generator = Path(self.config["path"]["generator"])
        self.generator = (
            (ROOT / generator).resolve() if not generator.is_absolute()
            else generator.resolve()
        )
        if not self.generator.is_file():
            raise FileNotFoundError(
                f"Configured path generator does not exist: {self.generator}"
            )
        self.output_dir = ROOT / "outputs" / self.experiment

    def _environment(self):
        environment = os.environ.copy()
        environment["MOTOR_META_EXPERIMENT"] = self.experiment
        environment["MOTOR_META_CONFIG"] = str(self.config_path)
        return environment

    def _script_for(self, stage):
        if stage not in STAGE_NAMES:
            raise ValueError(f"Unknown stage {stage!r}; choose from {STAGE_NAMES}")
        if stage == "path":
            return self.generator
        return SCRIPT_DIR / dict(STAGES)[stage]

    def run_stage(self, stage):
        """Run exactly one stage and return its completed process."""
        script = self._script_for(stage)
        print(f"\n=== {stage}: {script} ===", flush=True)
        return subprocess.run(
            [sys.executable, str(script)], cwd=ROOT,
            env=self._environment(), check=True,
        )

    def display_simulation(self):
        """Open the motion-review UI for motions produced by the motion stage."""
        print("\n=== motion-review: OpenSim visualizer ===", flush=True)
        return subprocess.run(
            [sys.executable, str(DISPLAY_SCRIPT)], cwd=ROOT,
            env=self._environment(), check=True,
        )

    def run_through(self, through="inference", start="path", display=None):
        """Run an inclusive consecutive range of stages."""
        if through not in STAGE_NAMES or start not in STAGE_NAMES:
            raise ValueError(f"Stages must be chosen from {STAGE_NAMES}")
        first, last = STAGE_NAMES.index(start), STAGE_NAMES.index(through)
        if first > last:
            raise ValueError("start must not come after through")
        print(f"Experiment outputs: {self.output_dir}", flush=True)
        show_motion = (
            self.config.get("display_simulation", False) if display is None else display
        )
        for stage in STAGE_NAMES[first:last + 1]:
            self.run_stage(stage)
            if stage == "motion" and show_motion:
                self.display_simulation()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--through", choices=STAGE_NAMES, default="inference")
    parser.add_argument(
        "--config", default="experiments/center_out.yaml",
        help="Experiment YAML containing the path generator and its parameters",
    )
    parser.add_argument(
        "--experiment", default=None,
        help="Optional override for the experiment name in the YAML",
    )
    args = parser.parse_args(argv)
    try:
        Pipeline(args.config, args.experiment).run_through(args.through)
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
