#!/bin/bash
# Regenerate all T99p (tas 0.99_exp_agg_NOPE_membspec, block 5) statistics and figures after completing the bootstrap seeds
#SBATCH --job-name="T99p_figs"
#SBATCH --output=temp/skill/%x_%j.out
#SBATCH --error=temp/skill/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --time=02:00:00

module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
set -x
M=0.99_exp_agg_NOPE_membspec
python -u -W ignore 17_bootstrap_skill_stats.py tas $M 5 100
python -u -W ignore plot_04_Fig3_uncertainty_skill.py tas $M 5 100 1
python -u -W ignore plot_04_Fig3_uncertainty_ag.py tas $M 5 100 1
python -u -W ignore plot_02_Fig2_bootstrap_uncertainty_summary.py tas $M 5 2025
python -u -W ignore plot_06_review_bootstrap_skill.py tas $M 5 100
