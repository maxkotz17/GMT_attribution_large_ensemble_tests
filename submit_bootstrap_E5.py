import pandas as pd
import numpy as np
import os
import glob

W=5
N=100

seeds=[s for s in range(N)]

for s, seed in enumerate(seeds):

	with open('slurm_example.sh', 'r') as file:
	    lines = file.readlines()

	lines[4]="#SBATCH --cpus-per-task=20 \n"
	lines[5]="#SBATCH --ntasks=1 \n"

	lines[8]="python -u bootstrap_ERA5.py tas 0.99 " + str(seed) + " " + str(W) + " \n"

	with open('submission_scripts/slurm_' + str(s) +'.sh', 'w') as file:
		file.writelines(lines)

	os.system("sbatch -A bsc32 -q bsc_es submission_scripts/slurm_" + str(s) + ".sh")

