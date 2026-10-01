#!/bin/bash
# Train the MyoSuite spindle decoder. One GPU, no specific node.
# mind-1-11 is TITAN Xp (Pascal); current PyTorch builds do not run there.
#
# Interactive form of this request:
#   srun -p gpu --cpus-per-task=4 --gres=gpu:1 --mem=64GB --time=12:00:00 --exclude=mind-1-11 --pty bash
#
# The trainer loads all 30,000 trials into RAM, so 10 GB is not enough.
#
#SBATCH --job-name=spindle-decoder
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --exclude=mind-1-11
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
# shellcheck disable=SC1091
source cluster/job_env.sh
activate_env

python train_decoder.py \
  --data "$DATA_FILE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --experiment-id "$EXPERIMENT_ID" \
  --training-seed 9 \
  --coefficient-seed "$SEED"
