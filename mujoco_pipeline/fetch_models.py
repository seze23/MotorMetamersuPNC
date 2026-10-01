"""Download the MS-Human-700 MuJoCo model used by this pipeline.

MyoSuite's arm is installed with the ``myosuite`` package. MS-Human-700 is
the separate full-body model from MuJoCo Menagerie and is not on PyPI.
"""

import subprocess
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_DIR / "third_party"
MENAGERIE = THIRD_PARTY / "mujoco_menagerie"
MANIPULATION_XML = MENAGERIE / "ms_human_700" / "MS-Human-700-Manipulation.xml"


def fetch_ms_human_700():
    """Clone MuJoCo Menagerie and check out only the MS-Human-700 model."""
    if MANIPULATION_XML.is_file():
        print(f"MS-Human-700 already present at {MANIPULATION_XML}")
        return MANIPULATION_XML
    THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    if not (MENAGERIE / ".git").exists():
        subprocess.run(
            [
                "git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                "https://github.com/google-deepmind/mujoco_menagerie.git",
                str(MENAGERIE),
            ],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(MENAGERIE), "sparse-checkout", "set", "ms_human_700"],
        check=True,
    )
    if not MANIPULATION_XML.is_file():
        raise FileNotFoundError(
            f"Sparse checkout finished but {MANIPULATION_XML} is missing."
        )
    print(f"Downloaded MS-Human-700 to {MANIPULATION_XML}")
    return MANIPULATION_XML


if __name__ == "__main__":
    fetch_ms_human_700()
