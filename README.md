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

Set `display_simulation: true` in an experiment YAML to pause after motion-file
generation and open the OpenSim model viewer. The accompanying control window
stays above the 3-D window and lets you select a generated path from the dropdown
or with Previous/Next, replay it, or continue the pipeline. The 3-D clock shows
elapsed time while the control window shows elapsed / total time. Playback speed,
preview frame rate, and the display-only floor height are configured under
`visualization`. The model's
referenced display meshes are stored beside the `.osim` file under `Geometry/`;
they affect visualization only.

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

## Stages

Pipeline code is organized independently of the original center-out experiment:

```text
process/
  generate_paths/
    generatereachpath.py
    draw.py
  utils/
  paths.py
  experiment.py
  ikcenterout.py
  gencenterout.py
  extractcenterout.py
  computefrcenterout.py
  centeroutinference.py
```

`generate_paths/` contains interchangeable path producers. The remaining files
consume the experiment manifest and therefore do not need to know which path
generator was used.

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
generator to `process/generate_paths/draw.py` will open a GUI in which paths can be
drawn and named interactively. Drawn curves are traversed once and held at their
endpoint by default. Set `path.return_to_rest: true` to retrace each curve back
to the resting hand. The GUI's average-hand-speed field is measured in cm/s and
is saved separately for each path; traversal duration is curve length / speed.

A generator may create any number of paths and choose their names at runtime.
It must write one `.npz` per path beneath the experiment's `paths/` directory,
then create `manifest.yaml` with `create_manifest()` from
`process.experiment`. Each path artifact must contain:

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

To implement another generator, use `process/generate_paths/generatereachpath.py` as a compact
reference: load `CONFIG`, build `xyz` and `times`, save beneath `PATHS_DIR`,
validate each file, and finally create the manifest. Do not hard-code output
directories, path names, sample counts, phase boundaries, or sampling rates.

## Training a model

Training is a separate workflow from `run_pipeline.py`. The simplified branch
contains the data conversion, dataset, network, training, and inference helper
modules, but it does not contain the original large raw training HDF5 file or a
one-command training launcher.

The relevant files have these roles:

- `extract_data/generate_train_test_data.py` converts biomechanical simulation
  data into spindle firing-rate inputs and seven-value trajectory labels.
- `extract_data/configs/train_test_data_spindles_extended.yaml` defines the
  muscles, optimal fiber lengths, spindle coefficients, sample rate, temporal
  length, and Ia/II afferent count used during conversion.
- `train/new_spindle_dataset.py` loads the converted HDF5 file and makes a
  deterministic 90% training / 10% validation split. It loads the entire file
  into memory.
- `model/model_definitions.py` defines the spatiotemporal CNN and its causal
  variant.
- `train/train_model_utils.py` provides `Trainer`, normalization, checkpointing,
  validation, early stopping, and `config.yaml` generation. It is a library,
  not an executable training script.
- `inference/test_model_utils_new.py` reconstructs a model from `config.yaml`,
  loads `model.ckpt`, and evaluates it. It is also a library.
- `process/centeroutinference.py` is the pipeline's executable inference stage.

### 1. Supply the raw biomechanical training data

The converter expects one HDF5 file with these datasets:

```text
muscle_lengths       (trials, 25, time)
muscle_velocities    (trials, 25, time)
muscle_accelerations (trials, 25, time)
endeffector_coords   (trials, 3, time)
joint_coords         (trials, 4, time)
```

Lengths, velocities, muscle ordering, coordinate conventions, and joint order
must match the YAML and the inference pipeline. The raw training file is not
included in this branch; outputs from a few path experiments are not enough to
reproduce the original 30,000-trial training set.

### 2. Convert lengths into spindle inputs

Review the extraction YAML before running this. Its default output has 5 Ia and
5 II channels for each of 25 muscles, 1,152 samples at 240 Hz, and labels
ordered as wrist XYZ followed by four joint angles.

```powershell
python -m extract_data.generate_train_test_data `
  --config_path extract_data/configs/train_test_data_spindles_extended.yaml `
  --input_file C:/path/to/raw_training_data.hdf5 `
  --output_dir C:/path/to/processed_data `
  --seeds 0 `
  --n_aff 5
```

The result contains:

```text
data    (trials, 10, 25, 1152)
labels  (trials, 1152, 7)
```

The converter currently caps processing at 30,000 trials and skips an output
file if it already exists. `--seeds` changes which spindle coefficient samples
are selected; it does not split the dataset.

### 3. Construct and train the network

Create a small launcher that instantiates the provided classes. This example
matches the architecture and naming convention expected by the current
pipeline checkpoint:

```python
import torch

from model.model_definitions import SpatiotemporalNetworkCausal
from train.new_spindle_dataset import SpindleDataset
from train.train_model_utils import Trainer

dataset = SpindleDataset(
    r"C:\path\to\processed_data\optimized_linear_extended_0_5_5_data.hdf5",
    dataset_type="train",
    task="letter_reconstruction_joints",
    n_out_time=1152,
)

model = SpatiotemporalNetworkCausal(
    experiment_id="causal_flag-pcr_optimized_linear_extended_5_5_letter_reconstruction_joints",
    nclasses=7,
    arch_type="spatiotemporal",
    nlayers=4,
    n_skernels=[8, 8, 32, 64],
    n_tkernels=[8, 8, 32, 64],
    s_kernelsize=7,
    t_kernelsize=7,
    s_stride=1,
    t_stride=1,
    padding=3,
    input_shape=[10, 25, 1152],
    p_drop=0.0,
    seed=0,
    training_seed=9,
    task="letter_reconstruction_joints",
    outtime=1152,
    my_dir="trained_models",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = Trainer(model=model, dataset=dataset, device=device)
trainer.train(
    num_epochs=100,
    learning_rate=5e-4,
    batch_size=256,
    val_steps=100,
    normalize=True,
)
```

Save that example as a script at the repository root and run it from there so
the package imports resolve. Training writes `model.ckpt`, `config.yaml`, and a
training plot beneath the model directory in `trained_models/`. GPU training is
strongly recommended for a full dataset.

Architecture, input shape, output count, task, normalization, afferent count,
muscle order, spindle seed, and training seed all become part of checkpoint
compatibility. If any of these change, do not point inference at the old
checkpoint.

### 4. Run inference with the trained checkpoint

Set `MODEL_PATH` in `process/centeroutinference.py` to the directory containing
the new `config.yaml` and `model.ckpt`, then run the normal pipeline:

```powershell
python run_pipeline.py --config experiments/center_out.yaml
```

To rerun only inference after earlier stages already populated the manifest:

```powershell
$env:MOTOR_META_CONFIG = (Resolve-Path experiments/center_out.yaml).Path
$env:MOTOR_META_EXPERIMENT = "center_out"
python process/centeroutinference.py
```

The pipeline inference stage builds an in-memory-compatible test HDF5 for each
trajectory, loads it with `SpindleDataset`, reconstructs the causal network from
the checkpoint configuration, and writes predictions and figures under the
experiment output directory.

## Changing the OpenSim model

Set `MODEL_PATH` in `process/paths.py`. A replacement model must
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
