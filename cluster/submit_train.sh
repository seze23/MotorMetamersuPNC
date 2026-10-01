#!/bin/bash
# Submit data generation, then training after that job succeeds.
# Run this from the repository root:
#   bash cluster/submit_train.sh
#
# Mind cluster: partitions are already cpu and gpu, and no account or
# node list is set. Before the first submission, run cluster/setup_env.sh
# once on a compute node so the conda env motor-metamers-mujoco exists.
# Trial count and epoch count live in cluster/job_env.sh.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

GENERATE_JOB=$(sbatch --parsable cluster/generate_reaches.sbatch)
echo "Generation job: $GENERATE_JOB"
TRAIN_JOB=$(sbatch --parsable --dependency="afterok:${GENERATE_JOB}" cluster/train_decoder.sbatch)
echo "Training job:   $TRAIN_JOB (starts after generation succeeds)"
