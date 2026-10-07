#!/bin/bash
# Run several 07_bootstrap seeds in parallel within one job (one python process per seed, ~3.5 GB each)
# Usage: sbatch ... run_bs_seeds_parallel.sh MODEL VARN METRIC W AGG MEMBSPEC SEED1 [SEED2 ...]
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --time=06:00:00

module load CONDA-FORGE/miniforge3-23.3.1-1
eval "$(conda shell.bash hook)"
conda activate climate
model=$1; varn=$2; metric=$3; W=$4; agg=$5; membspec=$6; shift 6
echo "model=$model varn=$varn metric=$metric W=$W agg=$agg membspec=$membspec seeds=$* host=$(hostname)"
for s in "$@"; do
	python -u -W ignore 07_bootstrap_ensemble_members.py $model $varn $metric $s 2 $W $agg $membspec > ${SLURM_SUBMIT_DIR}/temp/bs_pr099_agg/${model}_${varn}_agg${agg}_W${W}_seed${s}.log 2>&1 &
done
wait
echo "all seeds finished"
