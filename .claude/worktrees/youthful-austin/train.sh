#!/bin/bash
#$ -cwd
#$ -j y                     # join stdout and stderr
#$ -N gdr2-train          # job name
#$ -pe smp 12                # 8 cores per node
#$ -l h_rt=120:0:0           # 1 hour runtime
#$ -l h_vmem=7.5G           # 7.5G RAM per core
#$ -l gpu=1                # 4 GPUs
#$ -l gpu_type=ampere       # Ampere GPU (A100)
#$ -l cluster=andrena

#$ -o log.log


# Load environment
# module load miniforge
# mamba activate GDR
touch log.log
conda activate GDR
python --version

# Print some info
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $JOB_ID"
echo "Running on host: $(hostname)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Run with test config (minimal, fast)
python train.py
