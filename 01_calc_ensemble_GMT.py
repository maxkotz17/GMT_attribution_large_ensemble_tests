import pandas as pd
import numpy as np
import xarray as xr
import sys
import os
import glob

mod_paths=np.load("model_paths.npy")
mod_membs=np.load("model_members.npy",allow_pickle=True)

model=sys.argv[1]
#model="MPI-ESM1-2-LR"
path=mod_paths[np.where(np.char.find(mod_paths,"/"+model+"/")>0)][0]
members=mod_membs[np.where(np.char.find(mod_paths,"/"+model+"/")>0)][0]
abs0=273.15
ssp="ssp585"
ssp_pth=path.replace("historical",ssp).replace("CMIP/","ScenarioMIP/")
#limit members to those also in ssp, which seems to be one fewer for MPI
members=[x.split("/")[-1] for x in glob.glob(ssp_pth+"*")]

GMT_out=xr.Dataset()
for m, member in enumerate(members):
	file_path=path+"/"+member+"/day/tas/gn/"
	file_path=file_path+os.listdir(file_path)[0]+"/"
	#files=np.sort(os.listdir(file_path))
	files=np.sort(glob.glob(file_path+"*"))
	ssp_path=ssp_pth+member+"/day/tas/gn/"
	ssp_path=ssp_path+os.listdir(ssp_path)[0]+"/"
	ssp_files=np.sort(os.listdir(ssp_path))
	ssp_files=np.sort(glob.glob(ssp_path+"*"))
	if model=="CanESM5":
		ssp_files=[s for s in ssp_files if ".html" not in s]
		files=[f for f in files if ".html" not in f]
	first=True
	for f, filen in enumerate(list(files)+list(ssp_files)):
		yrend=int(filen[-11:-7])
		yrstart=int(filen[-20:-16])
		#load files from 1940 onwards, and up until 2100
		if (yrend>1900)&(yrstart<2101):
			x=xr.open_dataset(filen).sel(time=slice("1900","2100"))	
			tas=x["tas"][x.time.dt.year>1900]
			tas=tas[tas.time.dt.year<2101]
			#calculate GMT
			weights = np.cos(np.deg2rad(x.lat))
			weights.name = "weights"
			gmt=tas.weighted(weights).mean(("lon","lat"))
			gmt=gmt.groupby(gmt.time.dt.year).mean()-abs0
			if first:
				GMT=gmt
				first=False
			else:
				GMT=xr.concat((GMT,gmt),dim="year")	
	
		print(filen)

	GMT_out[member]=GMT

	print("DONE MEMBER: " + member) 
	print(str(m) + "/" + str(len(members)))

GMT_out=GMT_out.to_array(dim="member")
GMT_out=GMT_out.to_dataset(name="GMT")
GMT_out.to_netcdf("GMT_data/historical/" + model + "/" + model + "_GMT_historical_" + ssp + "_allmembers.nc")


