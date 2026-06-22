import pandas as pd
import numpy as np 
import xarray as xr
import sys
import os

mod_paths=np.load("model_paths.npy")
mod_membs=np.load("model_members.npy",allow_pickle=True)

model=sys.argv[1]
varn=sys.argv[2]
#model="MPI-ESM1-2-LR"
path=mod_paths[np.where(np.char.find(mod_paths,"/"+model+"/")>0)][0]
members=mod_membs[np.where(np.char.find(mod_paths,"/"+model+"/")>0)][0]
abs0=273.15

percile_output=xr.Dataset()
first=True
for m, member in enumerate(members):
	file_path=path+"/"+member+"/day/" + varn + "/gn/"
	file_path=file_path+os.listdir(file_path)[0]+"/"
	files=np.sort(os.listdir(file_path))

	first=True
	for f, filen in enumerate(files):
		#load files from 1940 onwards
		if (int(filen[-11:-7])>1939) & (int(filen[-20:-16])<1971):
			x=xr.open_dataset(file_path+filen)
			if first:
				tas=x[varn][(x.time.dt.year>1939) & (x.time.dt.year<1971)]
				first=False
			else:
				tas=xr.concat((tas,x[varn][(x.time.dt.year>1939) & (x.time.dt.year<1971)]),dim="time")		

	if varn=="tas":
		tas-=abs0
	elif varn=="pr":
		tas*=24*60*60
	prctile=tas.quantile([0.9,0.95,0.99],dim="time")
	percile_output[varn+"_"+member]=prctile
		
	print("Done " + member)
	print(str(m) + "/" + str(len(members)))

percile_output=percile_output.to_array(dim="member")
percile_output.name=varn + "_quantile"
percile_output.to_netcdf("thresholds/" + varn + "/" + model + "_" + varn + "1940-1970.nc")

