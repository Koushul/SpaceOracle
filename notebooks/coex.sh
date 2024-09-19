#!/bin/bash
#SBATCH --job-name=grnboost2
#SBATCH --output=coex.txt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40G
#SBATCH --time=2-00:00:00

source activate bee

python coex.py