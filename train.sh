#!/usr/bin/env bash
#SBATCH --job-name=dyno_paper_train
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/home/acw749/Dyno/logs/slurm-%x-%j.out
#SBATCH --error=/data/home/acw749/Dyno/logs/slurm-%x-%j.err

set -euo pipefail

EXPERIMENT="${EXPERIMENT:-paper_muq_1hz}"
SEED="${SEED:-0}"

cd /data/home/acw749/Dyno
mkdir -p logs

if command -v module >/dev/null 2>&1; then
  module load cuda/12.6.2-gcc-12.2.0
elif [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
  module load cuda/12.6.2-gcc-12.2.0
fi

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=dyno-paper

exec /data/home/acw749/conda-envs/dyno/bin/python -m dyno.train \
  "experiment=${EXPERIMENT}" \
  "seed=${SEED}" \
  trainer.devices=1 \
  trainer.strategy=auto \
  "$@"
