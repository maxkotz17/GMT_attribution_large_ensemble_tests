import pandas as pd
import numpy as np 
import xarray as xr
import sys
import os
import glob

varn="tas"
path="/esarchive/recon/ecmwf/era5/daily_mean/tas_f1h-r1440x721cds/"
abs0=273.15
files=np.sort(os.listdir(path))

threshold=xr.open_dataset("thresholds/"+varn+"/ERA5_"+varn+"1940-1970.nc")
threshold=threshold["tas"][-1,:,:]
threshs=threshold["quantile"].values
first=True
for f, filen in enumerate(files):
	#load files from 1940 onwards
	if (int(filen[-9:-5])>1939) & (int(filen[-9:-5])<2025):
		x=xr.open_dataset(path + filen)
		tas=x[varn][(x.time.dt.year>1939)]-abs0
		del x
		#aggregate months to year
		if f%12==0:
			tasyr=tas
		elif f>0:
			tasyr=xr.concat((tasyr,tas),dim="time")
		del tas
		#last month of year, calculate annual metric
		if f%12==11:
			metric=(tasyr>threshold).groupby(tasyr.time.dt.year).sum(dim="time")
			del tasyr
			if first:
				output=metric 
			else:
				output=xr.concat((output,metric),dim="year")
			first=False

	print(filen)

output.name=varn + "_exp"
output.to_netcdf("clim_extremes/" + varn + "/ERA5_" + varn + "_0.99_exp.nc")


