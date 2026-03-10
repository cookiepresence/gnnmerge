#!/bin/bash

#SBATCH -A research
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=gnnmerge-sweep
#SBATCH -o .sbatch-training-logs.out.%j
#SBATCH -e .sbatch-training-logs.err.%j
#SBATCH --time=2-00:00:00
#SBATCH --gpus=1
#SBATCH --mem=16GB

uv run python3 scripts/gnnmerge-sweep.py 2>&1 | tee sweep.log
