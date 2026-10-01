#!/bin/bash
# Build the MyoSuite reach file the decoder trains on. CPU only.
#
# Same flags as the Mind interactive example, without a node list:
#   srun -p cpu --cpus-per-task=8 --gres=gpu:0 --mem=16GB --time=4:00:00 --pty bash
#
#SBATCH --job-name=spindle-data
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:0
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs
# shellcheck disable=SC1091
source cluster/job_env.sh
activate_env

python -m mujoco_pipeline.generate_training_set \
  --backend "$BACKEND" \
  --num-trials "$NUM_TRIALS" \
  --workers "${SLURM_CPUS_PER_TASK:-$WORKERS}" \
  --seed "$SEED" \
  --output "$DATA_FILE"
