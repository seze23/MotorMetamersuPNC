import argparse
import ast
import os
import sys
import yaml
import torch
import h5py

from directory_paths import MODELS_DIR, SAVE_DIR
from model.model_definitions import SpatiotemporalNetworkCausal

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
# The pretrained model folder for the main phase-2 evaluation run.
# This should point at the model checkpoint architecture named spatiotemporal_4_8-8-32-64_7171.
MODEL_PATH = os.path.join(
    SAVE_DIR,
    MODELS_DIR,
    "experiment_causal_flag-pcr_optimized_linear_5_5_letter_reconstruction_joints",
    "spatiotemporal_4_8-8-32-64_7171_0_9",
)
TEST_HDF5 = os.path.join(SAVE_DIR, "data", "flag_pcr_test.hdf5")
TASK = "letter_reconstruction_joints"

# ──────────────────────────────────────────────────────────────────────────────
# Expected constants
# ──────────────────────────────────────────────────────────────────────────────
# Input is expected as [channels=10 afferent types] x [muscles=25] x [time=1152]
EXPECTED_INPUT_SHAPE = (10, 25, 1152)
# Output is the 7 joint-angle dimensions predicted by this model.
EXPECTED_OUTPUT_DIM = 7
EXPECTED_HOOK_CHANNELS = 64

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_config_value(value):
    """Convert a YAML string value into a native Python type.

    Args:
        value: A raw value read from the model config, often a string.

    Returns:
        The parsed Python object (int, float, bool, None, list, tuple, dict, or original string).

    Rationale:
        The training/config files in this project may encode hyperparameters as strings.
        This helper ensures the loaded PyTorch model is built with correct numeric types.
    """
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            pass
        low = text.lower()
        if low in {"true", "false", "none"}:
            return yaml.safe_load(text)
        try:
            if "." in text:
                return float(text)
            return int(text)
        except ValueError:
            return text
    return value


def load_config(model_path):
    """Load and normalize the YAML config for a pretrained model.

    Args:
        model_path: Directory path containing the saved model and its config.yaml.

    Returns:
        A dictionary of configuration values with parsed Python types.

    Raises:
        FileNotFoundError: if config.yaml does not exist.
        ValueError: if the config file format is not a dictionary.

    Rationale:
        Separating config loading from checkpoint loading is important for reproducible
        PyTorch model construction and for keeping this project aligned with the saved
        training metadata.
    """
    yaml_path = os.path.join(model_path, "config.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Unexpected config format in {yaml_path}")

    return {k: parse_config_value(v) for k, v in config.items()}


def select_device(force_cpu=False):
    """Select the best available device and print diagnostics."""
    if force_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[device] selected: {device}")
    if device.type == "cuda":
        print(f"[device] cuda available: {torch.cuda.is_available()}")
        print(f"[device] cuda version  : {torch.version.cuda}")
        print(f"[device] device count  : {torch.cuda.device_count()}")
        print(f"[device] device name   : {torch.cuda.get_device_name(0)}")

    return device


def build_model(config, model_path, device):
    """Instantiate the model architecture and load pretrained weights.

    Args:
        config: Parsed model configuration dictionary.
        model_path: Directory path containing the model checkpoint files.
        device: torch.device where the model should be loaded.

    Returns:
        A PyTorch model in evaluation mode with weights loaded.

    Rationale:
        Modular model construction plus checkpoint loading mirrors standard PyTorch
        packaging for reproducible inference in the project pipeline.
    """
    model = SpatiotemporalNetworkCausal(
        experiment_id=config["experiment_id"],
        nclasses=config["nclasses"],
        arch_type="spatiotemporal",
        nlayers=config["nlayers"],
        n_skernels=config["n_skernels"],
        n_tkernels=config["n_tkernels"],
        s_kernelsize=config["s_kernelsize"],
        t_kernelsize=config["t_kernelsize"],
        s_stride=config["s_stride"],
        t_stride=config["t_stride"],
        padding=config["padding"],
        input_shape=config["input_shape"],
        p_drop=config.get("p_drop", 0.0),
        seed=config.get("seed", None),
        train=True,
        task=TASK,
        outtime=config["outtime"],
        my_dir=SAVE_DIR,
        layer_norm=config.get("layer_norm", False),
    )

    for fname in ("model.pt", "model.pth", "best_model.pt", "best_model.pth", "model.ckpt"):
        ckpt_path = os.path.join(model_path, fname)
        if os.path.exists(ckpt_path):
            print(f"[load] Loading checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            elif isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state)
            break
    else:
        raise FileNotFoundError(
            f"No checkpoint found in {model_path}. Tried: model.pt, model.pth, best_model.pt, best_model.pth, model.ckpt"
        )

    model.to(device)
    model.eval()
    print(f"[load] Model on {device}, eval mode.")
    return model


_hook_store = {}


def get_hook(name="last_conv"):
    """Create a forward hook that captures intermediate activations.

    Args:
        name: The key under which to store the captured activation.

    Returns:
        A function compatible with torch.nn.Module.register_forward_hook.

    Rationale:
        Forward hooks enable inspection of intermediate feature maps without changing
        the model forward pass. This is useful for extracting the neural-space
        representation in the phase4 analysis pipeline.
    """
    def hook(module, input, output):
        _hook_store[name] = output.detach().cpu()

    return hook


def register_hook(model, layer_idx=3, name="last_conv"):
    """Attach a forward hook to a target layer in the model.

    Args:
        model: The PyTorch model instance.
        layer_idx: Index of the layer in model.model to hook.
        name: Label under which to save the captured activations.

    Returns:
        A hook handle that can be removed after inference.

    Rationale:
        Selecting a late convolutional layer here provides a consistent neural-space
        embedding for later activation comparisons while leaving the model code untouched.
    """
    handle = model.model[layer_idx].register_forward_hook(get_hook(name))
    print(f"[hook] Registered on model.model[{layer_idx}]: {model.model[layer_idx]}")
    return handle


def clear_activations():
    """Clear any previously captured hook activations from the global store.

    This helper keeps the shared hook storage clean between model evaluations.
    """
    _hook_store.clear()


def load_test_batch(hdf5_path, batch_size=4, device="cpu"):
    """Load one batch from the test HDF5 dataset and normalize tensor shapes.

    Args:
        hdf5_path: Path to the test HDF5 file containing spindle and joint data.
        batch_size: Number of samples to read.
        device: Torch device to move the tensors to.

    Returns:
        Tuple of (x, y) tensors where x is spindle input and y is joint-angle labels.

    Rationale:
        This function reuses the project data format to verify the main pretrained model
        on the same test split used in the phase2 evaluation pipeline.
    """
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as f:
        keys = list(f.keys())
        print("\n[data] HDF5 keys:", keys)

        if "spindle_info" in f and "joint_coords" in f:
            spindle = f["spindle_info"][:batch_size]
            joints = f["joint_coords"][:batch_size]
            print(f"[data] spindle_info raw shape : {spindle.shape}")
            print(f"[data] joint_coords raw shape : {joints.shape}")

            if spindle.ndim == 4 and spindle.shape[-1] == 2:
                raise ValueError(
                    "The loaded spindle_info appears to contain raw muscle-length/velocity pairs "
                    f"with shape {spindle.shape}. This script expects processed spindle afferent inputs "
                    f"with shape (B, 10, 25, 1152).\n\n"
                    "Please use the generated spindle dataset files, e.g. optimized_linear_0_5_5_flag_pcr_test.hdf5, "
                    "or run the data extraction pipeline from extract_data/ to convert raw muscle data into the 10-afferent input format."
                )
        elif "data" in f and "labels" in f:
            spindle = f["data"][:batch_size]
            joints = f["labels"][:batch_size]
            print(f"[data] data shape          : {spindle.shape}")
            print(f"[data] labels shape        : {joints.shape}")
        else:
            raise KeyError(
                f"Unsupported HDF5 layout for {hdf5_path}. Expected keys ['spindle_info','joint_coords'] "
                f"or ['data','labels'], got {keys}."
            )

    x = torch.tensor(spindle, dtype=torch.float32)
    y = torch.tensor(joints, dtype=torch.float32)

    if x.ndim == 3:
        x = x.unsqueeze(0)

    if x.shape[1:] != torch.Size(EXPECTED_INPUT_SHAPE):
        if x.ndim == 4 and x.shape[1] == EXPECTED_INPUT_SHAPE[2] and x.shape[2] == EXPECTED_INPUT_SHAPE[0] and x.shape[3] == EXPECTED_INPUT_SHAPE[1]:
            x = x.permute(0, 2, 3, 1)
        elif x.ndim == 4 and x.shape[1] == EXPECTED_INPUT_SHAPE[1] and x.shape[2] == EXPECTED_INPUT_SHAPE[0] and x.shape[3] == EXPECTED_INPUT_SHAPE[2]:
            x = x.permute(0, 2, 1, 3)

    if y.ndim == 3 and y.shape[-1] != EXPECTED_OUTPUT_DIM and y.shape[1] == EXPECTED_OUTPUT_DIM:
        y = y.permute(0, 2, 1)

    return x.to(device), y.to(device)


def main(model_path, test_hdf5, force_cpu=False):
    """Run the pretrained model verification pipeline.

    This script verifies that the saved phase2 model can be loaded, that a test
    batch from a processed spindle HDF5 is correctly shaped, and that both output and
    intermediate activations are produced as expected.

    Returns:
        A tuple (model, config) for potential downstream inspection.

    Rationale:
        Putting the full workflow in main ensures this file is a self-contained
        inference check and can be used as a sanity-test for the project pipeline.
    """
    device = select_device(force_cpu=force_cpu)
    print("\n" + "=" * 60)
    print("  load_pretrained.py — pipeline verification")
    print(f"  device: {device}")
    print("" + "=" * 60 + "\n")

    config = load_config(model_path)
    print(f"[config] experiment_id : {config['experiment_id']}")
    print(f"[config] input_shape   : {config['input_shape']}")
    print(f"[config] nclasses      : {config['nclasses']}")
    print(f"[config] outtime       : {config['outtime']}")

    model = build_model(config, model_path, device)
    print(f"\n[arch] model.model (conv blocks):\n{model.model}")
    print(f"\n[arch] fc_per_time_step: {model.fc_per_time_step}")

    hook_handle = register_hook(model, layer_idx=3, name="last_conv")

    x, y_true = load_test_batch(test_hdf5, batch_size=4, device=device)
    print(f"\n[data] Input tensor  x : {x.shape}  (expect B × 10 × 25 × 1152)")
    print(f"[data] Labels tensor y : {y_true.shape}")

    assert x.shape[1:] == torch.Size(EXPECTED_INPUT_SHAPE), (
        f"Input shape mismatch: got {x.shape[1:]}, expected {EXPECTED_INPUT_SHAPE}"
    )
    print("[check] Input shape matches expected (10, 25, 1152)")

    with torch.no_grad():
        y_pred, y_prob, net = model(x)

    print(f"\n[output] y_pred shape  : {y_pred.shape}  (expect B × 1152 × {EXPECTED_OUTPUT_DIM})")
    print(f"[hook]   activation shape: {_hook_store['last_conv'].shape}  (expect B × {EXPECTED_HOOK_CHANNELS} × 25 × 1152)")

    out_classes = y_pred.shape[-1]
    assert out_classes == EXPECTED_OUTPUT_DIM, (
        f"Output dim mismatch: got {y_pred.shape}, expected dim={EXPECTED_OUTPUT_DIM}"
    )
    print(f"[check] Output dim = {EXPECTED_OUTPUT_DIM} joint angles")

    hook_channels = _hook_store["last_conv"].shape[1]
    assert hook_channels == EXPECTED_HOOK_CHANNELS, (
        f"Hook channels mismatch: got {hook_channels}, expected {EXPECTED_HOOK_CHANNELS}"
    )
    print(f"[check] Hook activation channels = {EXPECTED_HOOK_CHANNELS}")

    print("\n" + "=" * 60)
    print("  SHAPE SUMMARY")
    print("=" * 60)
    print(f"  Input  (spindle_info) : {tuple(x.shape)}")
    print(f"  Output (joint angles) : {tuple(y_pred.shape)}")
    print(f"  Hook   (last conv)    : {tuple(_hook_store['last_conv'].shape)}")
    print(f"  Label  (joint_coords) : {tuple(y_true.shape)}")
    print("=" * 60)
    print(" Pipeline verification complete.\n")

    hook_handle.remove()

    return model, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a pretrained model and verify one test batch.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to the pretrained model directory containing config.yaml and checkpoint.",
    )
    parser.add_argument(
        "--test_hdf5",
        type=str,
        default=TEST_HDF5,
        help="Path to the HDF5 test file containing processed spindle afferent inputs.",
    )
    parser.add_argument(
        "--force_cpu",
        action="store_true",
        help="Force CPU execution even if CUDA is available.",
    )
    args = parser.parse_args()

    main(args.model_path, args.test_hdf5, force_cpu=args.force_cpu)
