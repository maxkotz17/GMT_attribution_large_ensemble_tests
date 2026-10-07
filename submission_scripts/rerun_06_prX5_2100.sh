#!/bin/bash
# Re-run linear LE attribution for PX5d to 2100 (MIROC6, MPI): June 5 files predate the % of baseline normalisation
#SBATCH --job-name="lin_prX5_2100"
#SBATCH --output=temp/skill/%x_%j.out
#SBATCH --error=temp/skill/%x_%j.err
#SBATCH --cpus-per-task=20
#SBATCH --ntasks=1
#SBATCH --time=02:00:00

module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
for m in MIROC6 MPI-ESM1-2-LR; do
	python -u -W ignore 06_attr_autocorr_lin_ens_membs.py $m pr X5 2100
done
