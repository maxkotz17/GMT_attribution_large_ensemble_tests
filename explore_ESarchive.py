import pandas as pd
import numpy as np
import os 
import sys
import glob

#CMIP6
file_path="/esarchive/exp/CMIP6/historical/"

pot_mods=glob.glob(file_path + "*")[1:]
mod_names=[x.split("/")[-1] for x in pot_mods]
group_names=[]
day_members=[]
no_members=[]
full_members=[]

for p, pot_mod in enumerate(pot_mods):
	group_name=glob.glob(pot_mod + "/CMIP/*")[0].split("/")[-1]
	group_names.append(group_name)	
	mod_name=mod_names[p]

	#get ensemble members
	members=glob.glob(pot_mod + "/CMIP/" + group_name + "/" + mod_name + "/historical/*")
	members=[x.split("/")[-1] for x in members]
	#filter by those with daily data
	day_members.append([])
	
	for m, member in enumerate(members):
		if glob.glob(pot_mod + "/CMIP/" + group_name + "/" + mod_name + "/historical/" + member + "/day"):
			day_members[p].append(member)
	
	no_members.append(len(day_members[p]))
	full_members.append(members)

mod_names=np.array(mod_names)
no_members=np.array(no_members)

mod_names[np.array(no_members)>10]
no_members[no_members>10]

paths=[pot_mod + "/CMIP/" + group_names[p] + "/" + mod_names[p] + "/historical/" for p, pot_mod in enumerate(pot_mods)]

np.save("model_paths.npy",paths)
np.save("model_members.npy",day_members,allow_pickle=True)

