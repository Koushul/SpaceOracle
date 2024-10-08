#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --job-name=SpaceOracle
#SBATCH --output=run.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00

# mamba deactivate
# mamba activate SpaceOracle

source activate bee
echo ${SLURM_JOB_NAME} allocated to ${SLURM_NODELIST}
echo environment $CONDA_DEFAULT_ENV
which python

python train.py
