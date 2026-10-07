#!/bin/bash
# Resubmit missing tas 0.99_exp membspec bootstrap seeds for a given block width
# Usage: sbatch -A bsc32 -q bsc_es -o temp/resub_tas099_bW/%x_%j.out -e temp/resub_tas099_bW/%x_%j.err resub_tas099_blockW.sh MODEL SEED W
#SBATCH --cpus-per-task=10
#SBATCH --ntasks=1
#SBATCH --time=12:00:00

module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
echo "model=$1 seed=$2 W=$3 host=$(hostname)"
python -u -W ignore 07_bootstrap_ensemble_members.py $1 tas 0.99_exp $2 2 $3 NOPE 1
