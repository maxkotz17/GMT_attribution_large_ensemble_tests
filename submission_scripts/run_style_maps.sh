#!/bin/bash
#SBATCH --job-name="stylemaps"
#SBATCH --output=temp/skill/%x_%j.out
#SBATCH --error=temp/skill/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
M=0.99_exp_agg_NOPE_membspec
# Uncertainty maps with chi/sigma label conventions (Figs 2, 4, A7, A10, A14 and sigma twins)
P="python -u -W ignore plot_02_Fig2_bootstrap_uncertainty_summary.py"
$P tas $M 5 2025
$P tas X5 5 none
$P pr $M 5 2025
$P pr X5 5 2025
$P pr X5 5 2100
$P pr $M 5 2025 sigma
$P pr X5 5 2025 sigma
