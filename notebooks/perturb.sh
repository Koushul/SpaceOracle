#!/bin/bash
#SBATCH --job-name=Bcl6
#SBATCH --output=perturb.txt
#SBATCH --ntasks=1                   
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G                   
#SBATCH --time=3-12:00:00               
#SBATCH --partition=htc

source activate bee
python perturb.py -goi Bcl6