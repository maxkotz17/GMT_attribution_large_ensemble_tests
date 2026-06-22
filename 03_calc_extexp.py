import pandas as pd
import numpy as np 
import xarray as xr
import sys
import os
import glob

mod_paths=np.load("model_paths.npy")
mod_membs=np.load("model_members.npy",allow_pickle=True)

model=sys.argv[1]
varn=sys.argv[2]
#Argument for whether to average thresholds across ensemble members
avens=bool(int(sys.argv[3]))
path=mod_paths[np.where(np.char.find(mod_paths,"/"+model+"/")>0)][0]
members=mod_membs[np.where(np.char.find(mod_paths,"/"+model+"/")>0)][0]
abs0=273.15
ssp="ssp585"
ssp_pth=path.replace("historical",ssp).replace("CMIP/","ScenarioMIP/")
#limit members to those also in ssp, which seems to be one fewer for MPI
members=[x.split("/")[-1] for x in glob.glob(ssp_pth+"*")]

threshold=xr.open_dataset("thresholds/"+varn+"/"+model+"_"+varn+"1940-1970.nc")
#threshold=threshold.mean(dim="member")
#select the 99%ile
t=-1
threshold=threshold[varn+"_quantile"][:,t,:,:]

member_output=xr.Dataset()
first=True
for m, member in enumerate(members):

	file_path=path+"/"+member+"/day/" + varn + "/gn/"
	file_path=file_path+os.listdir(file_path)[0]+"/"
	files=np.sort(glob.glob(file_path+"*.nc"))
	ssp_path=ssp_pth+member+"/day/" + varn + "/gn/"
	ssp_path=ssp_path+os.listdir(ssp_path)[0]+"/"
	ssp_files=np.sort(glob.glob(ssp_path+"*.nc"))

	if avens:
		thresh=thresh.mean(dim="member")
	else:
		thresh=threshold[m,...]

	first=True
	for f, filen in enumerate(list(files)+list(ssp_files)):
		yrend=int(filen[-11:-7])
		yrstart=int(filen[-20:-16])
		#load files from 1940 onwards
		if (yrend>1900)&(yrstart<2101):
			x=xr.open_dataset(filen)

			if varn=="tas":
				tas=x[varn][(x.time.dt.year>1900)]-abs0
			elif varn=="pr":
				tas=x[varn][(x.time.dt.year>1900)]*24*60*60
			else:
				print("unknown varn")
				break

			tas=tas[(tas.time.dt.year<2101)]
			metric=(tas>thresh).groupby(tas.time.dt.year).sum(dim="time")		

			if first:
				output=metric	                       
				first=False
			else:
				output=xr.concat((output,metric),dim="year")	

		print(filen)
	member_output[member]=output
	print("Done " + member)
	print(str(m) + "/" + str(len(members)))

threshs=[0.90,0.95,0.99]
member_output=member_output.to_array(dim="member")
member_output.name=varn + "_exp"
if avens:
	member_output.to_netcdf("clim_extremes/" + varn + "/" + model + "_" + varn + "_" + str(threshs[t]) + "_exp.nc")
else:
	member_output.to_netcdf("clim_extremes/" + varn + "/" + model + "_" + varn + "_" + str(threshs[t]) + "_exp_membspec.nc")

