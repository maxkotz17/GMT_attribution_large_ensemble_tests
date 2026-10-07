#!/bin/bash
# Statistics and remaining paper figures after completing T99p block 1/10 and aggregated P99p bootstraps
#SBATCH --job-name="remaining_figs"
#SBATCH --output=temp/skill/%x_%j.out
#SBATCH --error=temp/skill/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --time=03:00:00

module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
set -x
T=0.99_exp_agg_NOPE_membspec
#skill statistics
python -u -W ignore 17_bootstrap_skill_stats.py tas $T 1 100
python -u -W ignore 17_bootstrap_skill_stats.py tas $T 10 100
python -u -W ignore 17_bootstrap_skill_stats.py pr 0.99_exp_agg_5_membspec 5 200
#Figures A5, A6 (T99p block 1 / 10) and A12, A13 (P99p aggregated to ~5 / ~10 degrees) in the new Fig. 3/5 format
python -u -W ignore plot_04_Fig3_uncertainty_skill.py tas $T 1 100 0
python -u -W ignore plot_04_Fig3_uncertainty_skill.py tas $T 10 100 0
python -u -W ignore plot_04_Fig3_uncertainty_skill.py pr 0.99_exp_agg_5_membspec 5 200 0
python -u -W ignore plot_04_Fig3_uncertainty_skill.py pr 0.99_exp_agg_10_membspec 5 200 0
#previous-format versions for comparison
python -u -W ignore plot_04_Fig3_uncertainty_ag.py tas $T 1 100 0
python -u -W ignore plot_04_Fig3_uncertainty_ag.py tas $T 10 100 0
python -u -W ignore plot_04_Fig3_uncertainty_ag.py pr 0.99_exp_agg_5_membspec 5 200 0
python -u -W ignore plot_04_Fig3_uncertainty_ag.py pr 0.99_exp_agg_10_membspec 5 200 0
#Figures A10, A14 (PX5d maps to 2025 / 2100, block 5)
python -u -W ignore plot_02_Fig2_bootstrap_uncertainty_summary.py pr X5 5 2025
python -u -W ignore plot_02_Fig2_bootstrap_uncertainty_summary.py pr X5 5 2100
