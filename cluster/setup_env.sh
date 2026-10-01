#!/bin/bash
# One-time environment for the Mind cluster. Run it on a compute node,
# not on the login node mind.cs.cmu.edu:
#
#   srun -p cpu --cpus-per-task=4 --gres=gpu:0 --mem=16GB --time=1:00:00 --pty bash
#   bash cluster/setup_env.sh
#   exit
#
# Then, back on the login node: bash cluster/submit_train.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(hostname -s)" == "mind" ]]; then
  echo "This is the login node. Request a compute node first:"
  echo "  srun -p cpu --cpus-per-task=4 --gres=gpu:0 --mem=16GB --time=1:00:00 --pty bash"
  exit 1
fi

module load anaconda3
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n motor-metamers-mujoco python=3.11 -y
conda activate motor-metamers-mujoco
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r mujoco_pipeline/requirements.txt
pip install "myosuite==2.11.6"
python -c "import torch, mujoco, myosuite, h5py; print('ready', torch.__version__)"
