import pandas as pd
import numpy as np 
import xarray as xr
import sys
import os

varn="tas"
path="/esarchive/recon/ecmwf/era5/daily_mean/tas_f1h-r1440x721cds/"
abs0=273.15

percile_output=xr.Dataset()

files=np.sort(os.listdir(path))

first=True
for f, filen in enumerate(files):
	#load files from 1940 onwards
	if (int(filen[-9:-5])>1939) & (int(filen[-9:-5])<1971):
		x=xr.open_dataset(path + filen)
		if first:
			tas=x[varn][(x.time.dt.year>1939) & (x.time.dt.year<1971)]
			first=False
		else:
			tas=xr.concat((tas,x[varn][(x.time.dt.year>1939) & (x.time.dt.year<1971)]),dim="time")		
	print(filen)

tas-=abs0
prctile=tas.quantile([0.9,0.95,0.99],dim="time")
percile_output[varn]=prctile

percile_output.to_netcdf("thresholds/" + varn + "/ERA5_" + varn + "1940-1970.nc")

