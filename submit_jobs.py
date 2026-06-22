import pandas as pd
import numpy as np
import os
import glob

models=["CanESM5","MIROC6","MPI-ESM1-2-LR"]
varns=["pr"]
models=["CanESM5"]

for m, model in enumerate(models):
	for v, varn in enumerate(varns):

		with open('slurm_example.sh', 'r') as file:
		    lines = file.readlines()

		lines[4]="#SBATCH --cpus-per-task=20 \n"
		lines[5]="#SBATCH --ntasks=1 \n"

		lines[8]="python -u bootstrap_ensemble_members_lin.py  " + model + " " + varn +  " X5 1 2025 100 5 \n"

		with open('submission_scripts/slurm_' + str(m) +'.sh', 'w') as file:
			file.writelines(lines)

		os.system("sbatch -A bsc32 -q bsc_es submission_scripts/slurm_" + str(m) + ".sh")

