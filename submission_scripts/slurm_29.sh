#!/bin/bash
#SBATCH --job-name=2:29 
#SBATCH --output=temp/test%j.out
#SBATCH --error=temp/test%jerr
#SBATCH --cpus-per-task=20 
#SBATCH --ntasks=1 
#SBATCH --time=12:00:00

python -u attr_autocorr_poiss_ens_membs.py CanESM5 pr 0.99_exp 29 10 1 


