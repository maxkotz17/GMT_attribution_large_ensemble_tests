#!/bin/bash
#SBATCH --job-name="stylefix"
#SBATCH --output=temp/skill/%x_%j.out
#SBATCH --error=temp/skill/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
M=0.99_exp_agg_NOPE_membspec
# Skill figures with chi/sigma label conventions (Fig 3 variants, Fig 5, A17, Fig 6, A16)
P="python -u -W ignore plot_04_Fig3_uncertainty_skill.py"
$P tas X5 5 100 1
$P tas $M 5 100 1
$P tas $M 1 100 0
$P tas $M 10 100 0
$P pr $M 5 200 0
$P pr X5 5 200 0 2025
$P pr X5 5 200 0 2100
$P pr 0.99_exp_agg_5_membspec 5 200 0
$P pr 0.99_exp_agg_10_membspec 5 200 0
python -u -W ignore plot_05_Fig6_ERA5_attr_example.py
python -u -W ignore plot_uncertainty_breakdown.py tas X5 5
