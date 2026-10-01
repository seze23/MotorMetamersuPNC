# Shared settings for the Mind cluster jobs (mind.cs.cmu.edu).
# Edit this file, then from the login node run:
#   bash cluster/submit_train.sh
#
# BACKEND is myosuite or ms_human_700.
# NUM_TRIALS=30000 matches the original study size. The trainer loads the
# whole file, about 35 GB, so the GPU job asks for 64 GB of RAM.
# NUM_TRIALS=64 is a short smoke test.

BACKEND=myosuite
NUM_TRIALS=30000
WORKERS=8
SEED=0
EPOCHS=100
BATCH_SIZE=64
DATA_FILE="outputs_mujoco/training/${BACKEND}_${NUM_TRIALS}.hdf5"
EXPERIMENT_ID="mujoco_${BACKEND}_spindle_decoder"

# Created by cluster/setup_env.sh. Leave VENV empty when using conda.
CONDA_MODULE=anaconda3
CONDA_ENV=motor-metamers-mujoco

activate_env() {
  if [[ -n "${CONDA_ENV:-}" ]]; then
    module load "${CONDA_MODULE:-anaconda3}"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
  elif [[ -n "${VENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
  fi
}
