# Reproducing the horizontal-ellipse notebook

Main notebook: `notebooks/soechting_2005_horizontal_ellipses.ipynb`.
Source: Soechting & Poizner, 2005, DOI 10.1007/s00221-005-2316-8; supplied PDF in Downloads.
Configuration: `experiments/soechting_2005.yaml`.

The saved run contains 48 accepted full-size trials and 960 predictions from 20 matched archived checkpoint variants. The original 66-trial attempt included 18 rejected extreme-narrow trials. Their screening results are in `soechting_2005_rejected_input_audit.csv` and `outputs/soechting_2005/attempted_quality.csv`. Exclusion was based on muscle continuity before CNN predictions were inspected; the same shapes are excluded in every condition. A second rejected placement is archived under `outputs/soechting_2005_second_workspace_failed/`.

Run from repository root, in this order (or use the notebook's flags):

1. `.venv-opensim/bin/python -c "from run_pipeline import Pipeline; Pipeline('experiments/soechting_2005.yaml').run_stage('path')"`
2. `.venv-opensim/bin/python process/soechting_ik.py`
3. `.venv-opensim/bin/python -c "from run_pipeline import Pipeline; p=Pipeline('experiments/soechting_2005.yaml'); [p.run_stage(s) for s in ['motion','muscle-lengths','muscle-signals']]"`
4. `.venv-mujoco/bin/python process/soechting_analysis.py quality`
5. `.venv-mujoco/bin/python process/soechting_analysis.py inference`
6. `.venv-opensim/bin/python process/soechting_analysis.py readout`
7. `.venv-mujoco/bin/python -m process.soechting_report`

The environment named `mujoco` is used for Torch/plot libraries only; this experiment uses the OpenSim arm, not a MuJoCo replacement.

IK is an explicitly separate periodic phase-grid implementation using the same OpenSim model/couplers. Joint solutions have a posture prior, minimum shoulder elevation and periodic curvature regularization. No muscle-wrap surfaces, trained weights, or measured signals are edited. Interpolated IK is checked at every hand frame; fibers are then computed by the existing equilibrium extraction stage, not interpolated from the phase grid.

Geometry placement is an assumption because the paper does not report body-relative workspace and joint angles. Five-second preparatory transport and adaptive human decisions are omitted. The 0.25-s end buffer is an analysis protection, not part of the published protocol.

Generated artifact meanings:

- `quality.csv`: only the accepted inputs, with per-stage numerical screens. Passing is not physiological certification.
- `matched_spindles/`: five sets of matched synthetic spindle arrays.
- `spindle_quality.csv`: rate ranges and zero/upper-cap occupancy.
- `training_range.csv`: channel-standardized input deviations per checkpoint.
- `provenance.json`: configuration hash, every checkpoint hash, matching groups, padding/preprocessing.
- `variants/`: each variant's final-cycle regression outputs.
- `readouts/`: physical XYZ from each variant, actual hand positions and aligned times.
- `all_predictions.csv`: per-variant shape, error, center bias, fit quality and joint-limit screens.
- `ensemble_predictions.csv`: metrics of the pointwise mean physical trajectory, not mean metrics of individual variants.
- `model_pses.csv`: each variant's unique tested crossing and its bracket.
- `human_model_comparison.csv`: published/explicitly derived human aggregate means alongside model estimates. Human individual PSE data are unavailable; no PSE confidence bars are fabricated.

Experiment-1 human means (.021, .026, .026) are inferred from rounded pooled and experiment-2 means, not directly digitized participant data. Experiment-2 means (-.019, .030, -.002) are explicitly reported. Thresholds use 2 beta; available beta bootstrap intervals are doubled. Across-model SD and alpha-grid brackets are not human confidence intervals.

The notebook contains an asymmetric accepted alpha grid (-.25 to +1). It is not a complete reproduction of the two adaptive staircases; crossing interpolation is limited to tested values and not extrapolated. Small estimated speed-condition shifts inside the shared [0,.1] ensemble bracket remain unresolved.
