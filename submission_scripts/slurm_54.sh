#!/bin/bash
#SBATCH --job-name="par_test"
#SBATCH --output=temp/test%j.out
#SBATCH --error=temp/test%j.err
#SBATCH --cpus-per-task=20 
#SBATCH --ntasks=1 
#SBATCH --time=12:00:00

python -u bootstrap_ERA5.py tas 0.99 54 5 


