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

Set `MODEL_PATH` in `dataexp/centerout/paths.py`. A replacement model must
provide the following model-specific parameters, either under the current names
or through an adapter that maps them to this canonical interface.

### Required biomechanical parameters

- **Model file:** path to a valid OpenSim `.osim` model.
- **Reference marker:** the shoulder/reference origin (`R.Shoulder` currently).
- **End-effector marker:** the point tracked by IK (`Handle` currently). A fixed
  station or marker attached to the hand can be added if the model has none.
- **Elbow marker:** used only for saved elbow trajectories
  (`R.Elbow.Lateral` currently).
- **Rest pose:** one value in degrees for every coordinate needed to place the
  arm at the desired path origin. The current `REST` mapping is in
  `generatereachpath.py` and `ikcenterout.py`.
- **Driven coordinates:** coordinates set during muscle extraction. They are
  currently `elv_angle`, `shoulder_elv`, `shoulder_rot`, and `elbow_flexion`.
- **Motion-label coordinates:** ordered coordinates read from and written to
  `.mot` files. They are currently the four driven coordinates followed by
  `pro_sup`, `deviation`, and `flexion`.
- **Coordinate transform:** a 3-by-3 rotation from OpenSim ground coordinates
  to the pipeline world frame (`S2W`). Also specify position units; OpenSim uses
  meters while pipeline XYZ paths use shoulder-centered centimeters.
- **IK choices:** which coordinates are free, locked, or constrained; marker
  weights; joint limits; and any additional markers required to remove IK
  ambiguity. The current pipeline tracks only the end effector with weight 100.

### Required muscle and spindle parameters

- **Muscle list and order:** the exact ordered muscle names extracted by
  `extractcenterout.py`. The current neural input uses 25 muscles.
- **Optimal fiber lengths:** one value per muscle in
  `train_test_data_spindles_extended.yaml`, in the same order.
- **Ia and II spindle coefficients:** coefficient CSVs compatible with the
  selected muscles and ordered identically.
- **Sampling rate and sample count:** currently 240 Hz and 1,152 frames. Change
  downstream configuration if the replacement uses another temporal format.

### Required neural-model compatibility

The supplied CNN interface expects an input shaped
`(trial, 10 afferents, 25 muscles, time)`. A replacement musculoskeletal model
is checkpoint-compatible only if its muscles have a validated one-to-one
mapping into the same 25-muscle order and its signals use the same units,
normalization, spindle coefficients, sample rate, and label convention.
Renaming or reordering equivalent muscles can be handled by an adapter.
Changing the muscle count or physiological meaning generally requires new
coefficients and retraining the CNN.

### Replacement checklist

1. Load the model and verify all selected markers, coordinates, and muscles.
2. Define its rest pose and ground-to-world transform.
3. Confirm the rest end-effector round trip through the transform has near-zero
   error in `ikcenterout.py`.
4. Run one short path and inspect IK marker error and joint-limit violations.
5. Verify every extracted fiber length is finite, positive, and varies where
   expected.
6. Verify spindle ranges and velocities remain within the neural model's
   training distribution.
7. Only reuse a checkpoint after validating muscle ordering, normalization,
   temporal shape, and output-label semantics.
