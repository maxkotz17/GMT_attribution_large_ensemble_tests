#!/bin/bash
# Figures affected by renaming TX5d/PX5d -> T5d/P5d and by ranking sigma-map members by sigma correlation
#SBATCH --job-name="renfigs"
#SBATCH --output=temp/skill/%x_%j.out
#SBATCH --error=temp/skill/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
M=0.99_exp_agg_NOPE_membspec
python -u -W ignore plot_01_Fig1.py MIROC6 0.99_exp 1
python -u -W ignore plot_01_Fig1.py MIROC6 X5 0
python -u -W ignore plot_02_Fig2_bootstrap_uncertainty_summary.py pr $M 5 2025 sigma
python -u -W ignore plot_02_Fig2_bootstrap_uncertainty_summary.py pr X5 5 2025 sigma
python -u -W ignore plot_05_Fig5_precip_sigma_chi.py
python -u -W ignore plot_A17_sigma_chi.py
python -u -W ignore plot_05_Fig6_ERA5_attr_example.py
python -u -W ignore plot_uncertainty_breakdown.py tas X5 5
