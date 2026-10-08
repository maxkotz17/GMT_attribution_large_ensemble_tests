#!/bin/bash
# Cross-model benchmark of bootstrapped sigma maps (19_sigma_cross_model.py)
# Usage: sbatch -A bsc32 -q bsc_es run_sigma_cross.sh VARN METRIC W CUTOFF [ENDYR]
#SBATCH --job-name="sigcross"
#SBATCH --output=temp/skill/%x_%j.out
#SBATCH --error=temp/skill/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --time=02:00:00

module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
python -u -W ignore 19_sigma_cross_model.py $1 $2 $3 $4 $5
