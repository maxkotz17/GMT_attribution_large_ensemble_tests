#!/bin/bash
#SBATCH --job-name="par_test"
#SBATCH --output=temp/test%j.out
#SBATCH --error=temp/test%jerr
#SBATCH --cpus-per-task=5 
#SBATCH --ntasks=1 
#SBATCH --time=12:00:00

python -u attr_autocorr_lin_ens_membs.py MIROC6 pr X5 2025 

