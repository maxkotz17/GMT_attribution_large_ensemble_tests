import pandas as pd
import numpy as np
import os
import glob

models=["MPI-ESM1-2-LR","MIROC6","CanESM5"]

for m, model in enumerate(models):

	for mem in range(50):

		with open('slurm_example.sh', 'r') as file:
		    lines = file.readlines()

		lines[1]='#SBATCH --job-name=' + str(m) + ":" + str(mem) + ' \n'
		lines[4]="#SBATCH --cpus-per-task=20 \n"
		lines[5]="#SBATCH --ntasks=1 \n"

		lines[8]="python -u attr_autocorr_poiss_ens_membs.py " + model + " pr 0.99_exp " + str(mem) + " 10 1 \n"

		with open('submission_scripts/slurm_' + str(mem) +'.sh', 'w') as file:
			file.writelines(lines)

		os.system("sbatch -A bsc32 -q bsc_es submission_scripts/slurm_" + str(mem) + ".sh")

#for i in range(80):
#
#        with open('slurm_example.sh', 'r') as file:
#            lines = file.readlines()
#
#        lines[1]='#SBATCH --job-name=' + str(i) + ' \n'
#        lines[4]="#SBATCH --cpus-per-task=20 \n"
#        lines[5]="#SBATCH --ntasks=1 \n"
#
#        lines[8]="python -u attr_autocorr_poiss_ERA5.py tas 0.99_exp " + str(i) + " 80 \n"
#
#        with open('submission_scripts/slurm_' + str(i) +'.sh', 'w') as file:
#                file.writelines(lines)
#
#        os.system("sbatch -A bsc32 -q bsc_es submission_scripts/slurm_" + str(i) + ".sh")




