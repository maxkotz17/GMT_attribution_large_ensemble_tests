#!/bin/bash
# Resubmit missing tas 0.99_exp membspec block-5 bootstrap seeds (base spec)
# Usage: sbatch ... resub_tas099_block5.sh MODEL SEED
#SBATCH --job-name="bs_tas_b5"
#SBATCH --output=temp/resub_tas099_b5/%x_%j.out
#SBATCH --error=temp/resub_tas099_b5/%x_%j.err
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
echo "model=$1 seed=$2 host=$(hostname)"
python -u -W ignore 07_bootstrap_ensemble_members.py $1 tas 0.99_exp $2 2 5 NOPE 1
