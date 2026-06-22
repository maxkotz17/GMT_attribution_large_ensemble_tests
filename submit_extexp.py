import pandas as pd
import numpy as np
import os
import glob

models=["MIROC6","CanESM5","MPI-ESM1-2-LR"]
varns=["pr"]

for m, model in enumerate(models):
	for v, varn in enumerate(varns):

		with open('slurm_example.sh', 'r') as file:
		    lines = file.readlines()

		lines[4]="#SBATCH --cpus-per-task=20 \n"
		lines[5]="#SBATCH --ntasks=1 \n"

		lines[8]="python -u calc_TPX.py " + model + " " + varn + " 5 \n" 

		with open('submission_scripts/slurm_' + str(m) +'.sh', 'w') as file:
			file.writelines(lines)

		os.system("sbatch -A bsc32 -q bsc_es submission_scripts/slurm_" + str(m) + ".sh")

