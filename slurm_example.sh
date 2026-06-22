#!/bin/bash
#SBATCH --job-name="par_test"
#SBATCH --output=temp/test%j.out
#SBATCH --error=temp/test%jerr
#SBATCH --cpus-per-task=30
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

python -u bootstrap_ensemble_members_lin.py MPI-ESM1-2-LR pr X5 1 2100 100 5 

