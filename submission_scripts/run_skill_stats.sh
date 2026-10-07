#!/bin/bash
# Bootstrap skill statistics for the review (17_bootstrap_skill_stats.py)
# Usage: sbatch -A bsc32 -q bsc_es run_skill_stats.sh VARN METRIC W CUTOFF [ENDYR]
#SBATCH --job-name="skill"
#SBATCH --output=temp/skill/%x_%j.out
#SBATCH --error=temp/skill/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --time=02:00:00

module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
python -u -W ignore 17_bootstrap_skill_stats.py $1 $2 $3 $4 $5
