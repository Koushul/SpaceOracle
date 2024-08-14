#!/bin/bash
#SBATCH --job-name=sweep_cnn
#SBATCH --gres=gpu:1
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2-00:00:00 
#SBATCH --output=sweep_cnn.txt

cd ../../

source activate bee
# python -m spaceoracle.tools.sweep_vit
python -m spaceoracle.tools.sweep_cnn


