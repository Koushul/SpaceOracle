#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --job-name=SpaceOracle
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=300G
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-3:00:00

mamba deactivate
mamba activate SpaceOracle

python train.py