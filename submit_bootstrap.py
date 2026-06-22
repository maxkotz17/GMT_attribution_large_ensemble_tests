import pandas as pd
import numpy as np
import os
import glob

models=["MIROC6","MPI-ESM1-2-LR","CanESM5"]
Ns=[50]*3
W=2

for m, model in enumerate(models):

	N=Ns[m]
	seeds=[s for s in range(N)]
	#files=glob.glob("attribution/bootstrap/*" + model + "*block" + str(W) + "*seed*")
	#seeds_d=[int(f.split("seed")[1].split(".nc")[0]) for f in files]
	#seeds_l=list(set(seeds)-set(seeds_d))

	for s, seed in enumerate(seeds):

		with open('slurm_example.sh', 'r') as file:
		    lines = file.readlines()

		lines[4]="#SBATCH --cpus-per-task=10 \n"
		lines[5]="#SBATCH --ntasks=1 \n"

		lines[8]="python -u -W ignore bootstrap_ensemble_members.py " + model + " pr 0.99_exp " + str(seed) + " " + str(int(100/N)) + " " + str(W) + " 10 1 \n"

		with open('submission_scripts/slurm_' + str(m) +'.sh', 'w') as file:
			file.writelines(lines)

		os.system("sbatch -A bsc32 -q bsc_es submission_scripts/slurm_" + str(m) + ".sh")

