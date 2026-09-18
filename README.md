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

Run the default experiment described by `experiments/center_out.yaml`:

```powershell
python run_pipeline.py --config experiments/center_out.yaml
```

Use `--through muscle-signals` when the trained checkpoint is not available:

```powershell
python run_pipeline.py --config experiments/center_out.yaml --through muscle-signals
```

All generated files are isolated beneath the named experiment directory:

```text
outputs/
  center_out/
    manifest.yaml
    paths/
    ik/
    motions/
    muscles/
    spindles/
    predictions/
    figures/
  new_experiment_name/
    ...
```

For a new experiment, copy the YAML, give it a unique `experiment` value, and
change its path parameters or generator. `--experiment` can still override the
name without editing the YAML:

```powershell
python run_pipeline.py --config experiments/my_paths.yaml --through muscle-signals
```

Names may contain letters, numbers, underscores, and hyphens, and must begin
with a letter or number. Scripts run directly use `center_out` by default; set
the `MOTOR_META_EXPERIMENT` environment variable to select another directory.

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

## Path-generator contract

The generator is selected by `path.generator` in the experiment YAML. The
built-in `generatereachpath.py` reads all trajectory parameters from that YAML,
including sample rate, reach size, rest pose, and phase durations. Setting the
generator to `dataexp/centerout/draw.py` will open a GUI in which paths can be
drawn and named interactively (when that optional generator is installed).

A generator may create any number of paths and choose their names at runtime.
It must write one `.npz` per path beneath the experiment's `paths/` directory,
then create `manifest.yaml` with `create_manifest()` from
`dataexp.centerout.experiment`. Each path artifact must contain:

- `xyz`: a finite `(N, 3)` array in shoulder-centered world coordinates, in cm.
- `times`: a finite `(N,)` array in seconds, with strictly increasing values.
- Optional metadata such as `sample_rate_hz`, `center_world_cm`, and a path ID.

Call `validate_path_artifact()` before adding a path to the manifest. It checks
shape, time ordering, finite values, and the YAML's `max_displacement_cm` reach
limit. Paths may have different durations and sample rates; every downstream
stage now obtains timing and array sizes from the artifact instead of assuming
1,152 samples at 240 Hz. The current inference checkpoint may still require a
specific temporal distribution even though the pipeline itself does not.

The manifest is the handoff between stages. Downstream scripts process only its
listed artifacts and add their own entries (`ik_solution`, `motion`,
`muscle_data`, `spindle_data`, and predictions). This lets a generator decide
path count and names dynamically without predeclaring them in YAML.

To implement another generator, use `generatereachpath.py` as a compact
reference: load `CONFIG`, build `xyz` and `times`, save beneath `PATHS_DIR`,
validate each file, and finally create the manifest. Do not hard-code output
directories, path names, sample counts, phase boundaries, or sampling rates.

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
