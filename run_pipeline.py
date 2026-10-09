"""Run the simplified center-out pipeline in stage order.

The :class:`Pipeline` API is also used by ``run_pipeline.ipynb`` so the command
line and notebook interfaces execute exactly the same scripts.
"""

import argparse
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SCRIPT_DIR = ROOT / "process"
DISPLAY_SCRIPT = SCRIPT_DIR / "utils" / "display_sim.py"
MUJOCO_DISPLAY_SCRIPT = SCRIPT_DIR / "utils" / "display_mujoco.py"
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
        if self._model() == "myosuite_corrected":
            missing = [
                package for package in ("mujoco", "myo_sim")
                if importlib.util.find_spec(package) is None
            ]
            if missing:
                raise ModuleNotFoundError(
                    "myosuite_corrected requires missing package(s): "
                    f"{', '.join(missing)}. Update the active environment with "
                    "`conda env update -f environment.yml --prune`."
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
        if self._model() != "opensim":
            scripts = {
                "inverse-kinematics": "mujoco_ik.py",
                "motion": "mujoco_motion.py",
                "muscle-lengths": "mujoco_extract.py",
                "muscle-signals": "mujoco_spindles.py",
                "inference": "centeroutinference.py",
            }
            return SCRIPT_DIR / scripts[stage]
        return SCRIPT_DIR / dict(STAGES)[stage]

    def _model(self):
        name = str(self.config.get("musculoskeletal_model", "opensim")).lower()
        aliases = {
            "mobl_opensim": "opensim",
            "corrected_myoarm": "myosuite_corrected",
        }
        name = aliases.get(name, name)
        if name not in {"opensim", "myosuite_corrected"}:
            raise ValueError(
                "musculoskeletal_model must be 'opensim' or "
                "'myosuite_corrected'"
            )
        return name

    def _ik_backend(self):
        if self._model() != "opensim":
            return "mujoco"
        backend = str(self.config.get("ik", {}).get("backend", "opensim")).lower()
        if backend not in {"opensim", "nimble"}:
            raise ValueError("ik.backend must be either 'opensim' or 'nimble'")
        return backend

    @staticmethod
    def _wsl_path(path, distribution):
        path = Path(path).resolve()
        if not path.drive:
            raise ValueError(f"Cannot translate path to WSL: {path}")
        drive = path.drive.rstrip(":").lower()
        tail = path.as_posix().split(":", 1)[1].lstrip("/")
        return f"/mnt/{drive}/{tail}"

    def _nimble_command(self):
        options = self.config.get("ik", {}).get("nimble", {})
        distribution = options.get("wsl_distribution", "Ubuntu")
        python = options.get(
            "python", "/home/braydenk/.venvs/motor-meta-nimble/bin/python"
        )
        script = self._wsl_path(SCRIPT_DIR / "nimble_ik.py", distribution)
        config = self._wsl_path(self.config_path, distribution)
        return [
            "wsl", "-d", distribution, "--", "env",
            f"MOTOR_META_EXPERIMENT={self.experiment}",
            f"MOTOR_META_CONFIG={config}",
            python, script,
        ]

    def run_stage(self, stage):
        """Run exactly one stage and return its completed process."""
        if stage == "inverse-kinematics" and self._ik_backend() == "nimble":
            print("\n=== inverse-kinematics: Nimble Physics (WSL) ===", flush=True)
            return subprocess.run(self._nimble_command(), cwd=ROOT, check=True)
        script = self._script_for(stage)
        print(f"\n=== {stage}: {script} ===", flush=True)
        return subprocess.run(
            [sys.executable, str(script)], cwd=ROOT,
            env=self._environment(), check=True,
        )

    def display_simulation(self):
        """Open the model-appropriate viewer for the generated motions."""
        is_mujoco = self._model() != "opensim"
        viewer = MUJOCO_DISPLAY_SCRIPT if is_mujoco else DISPLAY_SCRIPT
        label = "MuJoCo" if is_mujoco else "OpenSim"
        print(f"\n=== motion-review: {label} visualizer ===", flush=True)
        return subprocess.run(
            [sys.executable, str(viewer)], cwd=ROOT,
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
