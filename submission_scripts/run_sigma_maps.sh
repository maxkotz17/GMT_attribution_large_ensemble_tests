#!/bin/bash
# Relative (Fig. 4 / A10) and absolute uncertainty maps for precipitation extremes (plot_02_Fig2_bootstrap_uncertainty_summary.py)
#SBATCH --job-name="sigmaps"
#SBATCH --output=temp/skill/%x_%j.out
#SBATCH --error=temp/skill/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
M=0.99_exp_agg_NOPE_membspec
python -u -W ignore plot_02_Fig2_bootstrap_uncertainty_summary.py pr $M 5 2025 sigma
python -u -W ignore plot_02_Fig2_bootstrap_uncertainty_summary.py pr X5 5 2025 sigma
