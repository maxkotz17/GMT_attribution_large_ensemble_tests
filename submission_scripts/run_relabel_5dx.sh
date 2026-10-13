#!/bin/bash
# Figures affected by renaming T5d/P5d -> T5dx/P5dx
#SBATCH --job-name="relabel"
#SBATCH --output=temp/skill/%x_%j.out
#SBATCH --error=temp/skill/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
python -u -W ignore plot_05_Fig5_precip_sigma_chi.py
python -u -W ignore plot_A17_sigma_chi.py
python -u -W ignore plot_A18_bootstrap_inference.py
python -u -W ignore plot_05_Fig6_ERA5_attr_example.py
python -u -W ignore plot_uncertainty_breakdown.py tas X5 5
