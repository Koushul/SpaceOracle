#!/bin/bash 
#SBATCH --job-name=sweep-2
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00 
#SBATCH --output=sweep-2.txt

source activate bee
cd ../../

python -m spaceoracle.tools.sweep_vit