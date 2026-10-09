"""Record MuJoCo qpos trajectories as the pipeline motion artifact."""

from pathlib import Path
import shutil

from experiment import load_manifest, resolve_artifact, set_artifact
from paths import MOTIONS_DIR


def main():
    trajectories = load_manifest()["trajectories"]
    for trajectory in trajectories:
        source = Path(resolve_artifact(trajectory["ik_solution"]))
        output = Path(MOTIONS_DIR) / f"{trajectory['id']}.npz"
        shutil.copyfile(source, output)
        set_artifact(trajectory["id"], "motion", output)
        print(f"  {trajectory['id']:<20} -> {output}")
    print(f"Generated {len(trajectories)} MuJoCo motion artifact(s).")


if __name__ == "__main__":
    main()
