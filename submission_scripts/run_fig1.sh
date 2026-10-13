#!/bin/bash
#SBATCH --job-name="fig1"
#SBATCH --output=temp/skill/%x_%j.out
#SBATCH --error=temp/skill/%x_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
for s in plot_01_Fig1.py; do
python -u -W ignore $s MIROC6 0.99_exp 1
python -u -W ignore $s MIROC6 X5 0
done
