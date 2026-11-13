# Local experiment artifacts

Generated results are intentionally excluded from Git. Existing results were
moved to `local_artifacts/` without deleting them:

- `local_artifacts/dataexp/`: result tables, generated arrays, and diagnostic figures
- `local_artifacts/outputs/`: generated kinematics, muscle, spindle, and model outputs
- `local_artifacts/docs_archive/`: superseded investigation handoffs
- `local_artifacts/code_archive/`: exploratory scripts and experiment configurations

New files under `dataexp/`, `outputs/`, and `local_artifacts/` are ignored. Files
already tracked historically under `dataexp/wang_grid/` and
`dataexp/wang_pilot/` remain tracked.

`scripts/build_pi_figures.py` reads `local_artifacts/dataexp/` when present and
falls back to `dataexp/`. Set `MOTOR_META_DATAEXP` to select another result root.

The archived material is local-only and is not available after a fresh clone.
Copy or back it up separately before removing the working directory.

