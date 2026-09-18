# Simplified center-out proprioception pipeline

This branch keeps only the code and data needed to turn a desired wrist path
into OpenSim joint motion, muscle-fiber lengths, spindle firing rates, and
(when weights are supplied) neural-network position estimates.

## Setup

```powershell
conda env create -f environment.yml
conda activate motor-meta-simplified
python -c "import opensim; print(opensim.GetVersionAndDate())"
```

Put the pretrained model files here before running stage 6:

```text
trained_models/
  experiment_causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints/
    spatiotemporal_4_8-8-32-64_7171_0_9/
      config.yaml
      model.ckpt
```

## Run

Run everything:

```powershell
python run_pipeline.py
```

Use `--through muscle-signals` when the trained checkpoint is not available:

```powershell
python run_pipeline.py --through muscle-signals
```

All generated files are written to `outputs/`.

The display meshes were intentionally omitted. OpenSim therefore prints
`Couldn't find file '*.vtp'` warnings while loading the model; these affect GUI
appearance only, not IK, muscle equilibrium, or spindle calculations.

## Stages

1. `generatereachpath.py`: define desired wrist XYZ samples.
2. `ikcenterout.py`: use OpenSim inverse kinematics to solve joint angles.
3. `gencenterout.py`: write the joint trajectory as OpenSim `.mot` files.
4. `extractcenterout.py`: pose the model and extract 25 muscle-fiber lengths.
5. `computefrcenterout.py`: convert length, velocity, and acceleration to Ia/II firing rates.
6. `centeroutinference.py`: use the pretrained CNN to predict wrist and joint state.

The trajectory is defined in `dataexp/centerout/generatereachpath.py`. To make
a new path, replace the generated `(N, 3)` `xyz` array while preserving units
(centimeters), coordinate convention, and a matching `times` array. The IK
stage consumes every `outputs/desired_xyz_*.npz` file automatically.

## Changing the OpenSim model

Set `MODEL_PATH` in `dataexp/centerout/paths.py`. A replacement must expose the
markers and coordinates used by the scripts (`R.Shoulder`, `Handle`,
`R.Elbow.Lateral`, and the seven named coordinates). If its names or degrees of
freedom differ, update the marker/coordinate constants in `ikcenterout.py` and
`extractcenterout.py`. Also update the 25-muscle list and spindle optimal
lengths if the muscle set changes. A neural checkpoint trained on the old
25-muscle input is not compatible with a different input muscle ordering or
count without an explicit mapping or retraining.
