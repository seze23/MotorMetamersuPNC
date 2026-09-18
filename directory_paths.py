from pathlib import Path

PARENT_DIR = str(Path(__file__).resolve().parent)
SAVE_DIR = PARENT_DIR
MODELS_DIR = str(Path(PARENT_DIR) / "trained_models")
