"""Efficiently run the 5 coefficient seeds x 4 matched checkpoints."""

import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "experiments" / "wang_grid.yaml"
RUNTIME_CONFIG_DIR = ROOT / "outputs" / "wang_grid" / "runtime_configs"
CHECKPOINT_ROOT = (
    ROOT / "trained_models"
    / "experiment_causal_flag-pcr_optimized_linear_extended_5_5_"
      "letter_reconstruction_joints"
)
TRAINING_SEEDS = (0, 1, 2, 9)


def run(script, config):
    environment = os.environ.copy()
    environment["MOTOR_META_EXPERIMENT"] = "wang_grid"
    environment["MOTOR_META_CONFIG"] = str(config)
    subprocess.run([sys.executable, str(ROOT / "process" / script)],
                   cwd=ROOT, env=environment, check=True)


def main():
    with BASE_CONFIG.open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    RUNTIME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for coefficient_seed in range(5):
        pending_training_seeds = [
            seed for seed in TRAINING_SEEDS
            if not (
                ROOT / "dataexp" / "wang_grid"
                / f"model_{coefficient_seed}_{seed}_bias.csv"
            ).is_file()
        ]
        if not pending_training_seeds:
            print(f"Skipping completed coefficient seed {coefficient_seed}.")
            continue
        coefficient_config = dict(base)
        coefficient_config["proprioception"] = dict(base["proprioception"])
        coefficient_config["proprioception"]["coefficient_seed"] = coefficient_seed
        coefficient_path = RUNTIME_CONFIG_DIR / f"coef_{coefficient_seed}.yaml"
        with coefficient_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(coefficient_config, stream, sort_keys=False)
        print(f"\n=== spindle coefficient seed {coefficient_seed} ===", flush=True)
        run("computefrcenterout.py", coefficient_path)

        for training_seed in pending_training_seeds:
            model_name = f"{coefficient_seed}_{training_seed}"
            config = dict(base)
            config["proprioception"] = dict(base["proprioception"])
            config["proprioception"].update({
                "coefficient_seed": coefficient_seed,
                "training_seed": training_seed,
                "model_path": str(
                    CHECKPOINT_ROOT
                    / f"spatiotemporal_4_8-8-32-64_7171_{model_name}"
                ),
            })
            config_path = RUNTIME_CONFIG_DIR / f"model_{model_name}.yaml"
            with config_path.open("w", encoding="utf-8") as stream:
                yaml.safe_dump(config, stream, sort_keys=False)
            print(f"\n--- matched model {model_name} ---", flush=True)
            run("batch_wang_grid_inference.py", config_path)
            run("analyze_wang_grid.py", config_path)
    print("Completed all 20 matched models.")


if __name__ == "__main__":
    main()
