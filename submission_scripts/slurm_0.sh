#!/bin/bash
#SBATCH --job-name="par_test"
#SBATCH --output=temp/test%j.out
#SBATCH --error=temp/test%jerr
#SBATCH --cpus-per-task=20 
#SBATCH --ntasks=1 
#SBATCH --time=12:00:00

python -u bootstrap_ensemble_members_lin.py  CanESM5 pr X5 1 2025 100 5 

