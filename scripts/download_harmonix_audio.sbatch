#!/usr/bin/env bash
#SBATCH --job-name=coverx-harmonix-audio
#SBATCH --output=/gpfs/scratch/acw749/datasets/structure/logs/slurm-%x-%j.out
#SBATCH --error=/gpfs/scratch/acw749/datasets/structure/logs/slurm-%x-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

cd /data/home/acw749/Dyno
export PYTHONPATH=/data/home/acw749/Dyno:${PYTHONPATH:-}
export COVERX_TOOLS_PATH=/gpfs/scratch/acw749/datasets/structure/tools/coverx_tools
# Optional: export COVERX_YOUTUBE_COOKIES=/gpfs/scratch/acw749/datasets/structure/secrets/youtube-cookies.txt

/data/home/acw749/conda-envs/instruct_embed/bin/python -m dyno.download_structure_data --download-harmonix-audio
