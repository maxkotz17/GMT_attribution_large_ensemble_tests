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
N=int(sys.argv[3])
#model="MPI-ESM1-2-LR"
path=mod_paths[np.where(np.char.find(mod_paths,"/"+model+"/")>0)][0]
members=mod_membs[np.where(np.char.find(mod_paths,"/"+model+"/")>0)][0]
abs0=273.15
ssp="ssp585"
ssp_pth=path.replace("historical",ssp).replace("CMIP/","ScenarioMIP/")
#limit members to those also in ssp, which seems to be one fewer for MPI
members=[x.split("/")[-1] for x in glob.glob(ssp_pth+"*")]

member_output=xr.Dataset()
first=True
for m, member in enumerate(members):

	file_path=path+"/"+member+"/day/" + varn + "/gn/"
	file_path=file_path+os.listdir(file_path)[0]+"/"
	files=np.sort(glob.glob(file_path+"*"))
	ssp_path=ssp_pth+member+"/day/" + varn + "/gn/"
	ssp_path=ssp_path+os.listdir(ssp_path)[0]+"/"
	ssp_files=np.sort(glob.glob(ssp_path+"*"))
	ssp_files=[ssp_file for ssp_file in ssp_files if ".html" not in ssp_file]

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
			del x
			tas=tas[(tas.time.dt.year<2101)]
			#Maximum annual value of rolling mean
			metric=tas.rolling(time=N).mean().groupby(tas.time.dt.year).max(dim="time")

			if first:
				output=metric	
				first=False
			else:
				output=xr.concat((output,metric),dim="year")	

		print(filen)
	member_output[member]=output
	print("Done " + member)
	print(str(m) + "/" + str(len(members)))

member_output=member_output.to_array(dim="member")
member_output.name=varn + "_X" +str(N)
member_output.to_netcdf("clim_extremes/" + varn + "/" + model + "_" + varn + "_X" + str(N) + ".nc")

