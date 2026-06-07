#!/usr/bin/env bash
#SBATCH -J struct_missing_sae
#SBATCH -p sae
#SBATCH -A pilot_sae_gpu
#SBATCH -n 1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=hopper|ampere
#SBATCH --mem-per-cpu=7500M
#SBATCH -t 12:00:00
#SBATCH -o /gpfs/scratch/acw749/datasets/structure/logs/slurm-%x_%j.out
#SBATCH -e /gpfs/scratch/acw749/datasets/structure/logs/slurm-%x_%j.err

set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load cuda/12.6.2-gcc-12.2.0
elif [[ -f /etc/profile.d/modules.sh ]]; then
  # Some Slurm batch shells do not initialize environment modules.
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
  module load cuda/12.6.2-gcc-12.2.0
else
  echo "[structure-extract] warning: environment modules unavailable; continuing without module load"
fi

/usr/bin/env bash /data/home/acw749/Dyno/scripts/extract_structure_missing_common.sh
