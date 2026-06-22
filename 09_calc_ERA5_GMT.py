import pandas as pd
import numpy as np 
import xarray as xr
import sys
import os
import glob

varn="tas"
path="/esarchive/recon/ecmwf/era5/daily_mean/tas_f1h-r1440x721cds/"
abs0=273.15
files=np.sort(glob.glob(path + varn + "*"))

first=True
for f, filen in enumerate(files):
	#load files from 1940 onwards
	if (int(filen[-9:-5])>1930) & (int(filen[-9:-5])<2026):
		x=xr.open_dataset(filen)
		tas=x[varn][(x.time.dt.year>1930)]-abs0

		weights = np.cos(np.deg2rad(tas.lat))
		weights.name = "weights"
		gmt=tas.weighted(weights).mean(("lon","lat"))

		del x
		#aggregate months to year
		if f%12==0:
			gmtyr=gmt
		elif f>0:
			gmtyr=xr.concat((gmtyr,gmt),dim="time")
		del tas
		#last month of year, calculate annual metric
		if f%12==11:
			metric=gmtyr.groupby(gmtyr.time.dt.year).mean()
			del gmtyr
			if first:
				output=metric 
			else:
				output=xr.concat((output,metric),dim="year")
			first=False

	print(filen)

output.name="GMT"
output.to_netcdf("GMT_data/ERA5_" + varn + ".nc")


