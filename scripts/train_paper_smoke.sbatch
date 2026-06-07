#!/usr/bin/env bash
#SBATCH --job-name=dyno_paper_smoke
#SBATCH --partition=andrena
#SBATCH --account=pilot_andrena
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --mem-per-cpu=7500M
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=/data/home/acw749/Dyno/logs/slurm-%x-%A_%a.out
#SBATCH --error=/data/home/acw749/Dyno/logs/slurm-%x-%A_%a.err

set -euo pipefail

EXPERIMENTS=(
  paper_muq_1hz
  paper_muq_1hz_no_anchor
  paper_mert_1hz
)
ALIASES=(
  smoke-muq-residual-vae-d32
  smoke-muq-no-anchor-d32
  smoke-mert-residual-vae-d32
)

INDEX="${SLURM_ARRAY_TASK_ID}"
EXPERIMENT="${EXPERIMENTS[$INDEX]}"
ALIAS="${ALIASES[$INDEX]}"

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

/data/home/acw749/conda-envs/dyno/bin/python -m dyno.train \
  "experiment=${EXPERIMENT}" \
  trainer.max_steps=210 \
  trainer.devices=1 \
  trainer.strategy=auto \
  data.num_workers=8 \
  "name=paper/smoke/${ALIAS}/seed0" \
  "tags=[paper,phase3_smoke,${ALIAS}]"
