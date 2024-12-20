#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --job-name=SpaceOracle
#SBATCH --output=kidney/train_logs/train.txt
#SBATCH --error=kidney/train_logs/train.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-5:00:00

source activate bee

python train.py