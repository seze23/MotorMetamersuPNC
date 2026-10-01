"""Train the spindle-to-pose decoder.

The decoder is SpatiotemporalNetworkCausal in model/model_definitions.py.
This script is the cluster entry point. Run it from any directory:

    python train_decoder.py --data outputs_mujoco/training/myosuite_30000.hdf5

The HDF5 must contain ``data`` (trials, 10, 25, time) and ``labels``
(trials, time, 7). Generate it with mujoco_pipeline.generate_training_set.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="HDF5 with data and labels datasets")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--val-steps", type=int, default=100)
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu. Default: cuda if available.")
    parser.add_argument("--experiment-id", default="mujoco_myosuite_spindle_decoder")
    parser.add_argument("--training-seed", type=int, default=9)
    parser.add_argument("--coefficient-seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        import torch
    except ImportError as error:
        raise SystemExit(
            "PyTorch is not installed in this environment. On the cluster, "
            "install the CUDA build that matches the GPU before submitting."
        ) from error

    from model.model_definitions import SpatiotemporalNetworkCausal
    from train.new_spindle_dataset import SpindleDataset
    from train.train_model_utils import Trainer

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    if not data_path.is_file():
        raise SystemExit(f"Training file not found: {data_path}")

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training on {device} from {data_path}", flush=True)

    dataset = SpindleDataset(
        str(data_path),
        dataset_type="train",
        task="letter_reconstruction_joints",
        n_out_time=1152,
    )
    print(
        f"Train samples {tuple(dataset.train_data.shape)}, "
        f"validation samples {tuple(dataset.val_data.shape)}",
        flush=True,
    )

    model = SpatiotemporalNetworkCausal(
        experiment_id=args.experiment_id,
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
        seed=args.coefficient_seed,
        training_seed=args.training_seed,
        task="letter_reconstruction_joints",
        outtime=1152,
        my_dir="trained_models",
    )
    print(f"Checkpoints: {model.model_path}", flush=True)
    trainer = Trainer(model=model, dataset=dataset, device=device)
    trainer.train(
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        val_steps=args.val_steps,
        normalize=True,
    )
    print(f"Finished. Weights: {Path(model.model_path) / 'model.ckpt'}", flush=True)


if __name__ == "__main__":
    main()
