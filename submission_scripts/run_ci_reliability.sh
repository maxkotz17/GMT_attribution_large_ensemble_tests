#!/bin/bash
# Coverage of bootstrap intervals and sign detection against the large ensemble (20_bootstrap_ci_reliability.py)
# Usage: sbatch -A bsc32 -q bsc_es run_ci_reliability.sh VARN METRIC W CUTOFF [ENDYR]
#SBATCH --job-name="cirel"
#SBATCH --output=temp/skill/%x_%j.out
#SBATCH --error=temp/skill/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --time=02:00:00

module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
python -u -W ignore 20_bootstrap_ci_reliability.py $1 $2 $3 $4 $5
