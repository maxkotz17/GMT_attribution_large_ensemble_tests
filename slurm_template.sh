#!/bin/bash
#SBATCH --job-name="ERA5_extr"
#SBATCH --output=temp/test%j.out
#SBATCH --error=temp/test%j.err
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

srun -n 1 python -u plot_uncertainty_breakdown.py tas X5 5 

