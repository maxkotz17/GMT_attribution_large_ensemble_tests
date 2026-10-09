import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from matplotlib import gridspec
import sys
from matplotlib.colors import TwoSlopeNorm
import glob
import cartopy.crs as ccrs
import string

varn="tas"
metric="X5"
W=5

#Load ERA5 GMT
GMT=xr.open_dataset("GMT_data/ERA5_tas.nc")
GMT=GMT["GMT"]
#Smooth with 9 year rolling mean and reflecting boundary conditions to preserve data up to present
GMT=xr.concat((GMT[::-1],GMT,GMT[::-1]),dim="year").rolling(year=9,center=True).mean()[GMT.shape[0]:2*GMT.shape[0]]
GMT=GMT[~np.isnan(GMT)]

#GMT difference from pre 1980 average
dGMT=GMT-GMT.loc[GMT.year<1980].mean()

#Load ERA5 TX5
obs=xr.open_dataset("clim_extremes/" + varn + "/ERA5_" + varn + "_" + metric + ".nc")
obs=obs.tas_X5
obs=obs.loc[obs.year.isin(GMT.year)]
# Convert obs lon from 0–360 to -180–180
obs = obs.assign_coords(lon=(obs.lon + 180) % 360 - 180).sortby("lon")

#Load bootstrap coefficients from ERA5
folder="/gpfs/scratch/bsc32/bsc400019/"
coefs_df=xr.open_dataset(folder + "attribution/bootstrap/attr_coefs_ERA5_" + varn + "_" + metric + "_block" + str(W) + ".nc").coef
coefs=xr.open_dataset(folder + "attribution/attr_coefs_ERA5_tas_X5.nc")

#Locations
city_dict = {
    "Madrid":        {"lat": 40.4168,  "lon": -3.7038},
    "Delhi":         {"lat": 28.6139,  "lon": 77.2090},
    "Riyadh":        {"lat": 24.7136,  "lon": 46.6753},
    "Berlin":        {"lat": 52.5200,  "lon": 13.4050},
    "Jakarta":       {"lat": -6.2088,  "lon": 106.8456},
    "Sydney":        {"lat": -33.8688, "lon": 151.2093},
    "São Paulo":     {"lat": -23.5505, "lon": -46.6333},
    "Johannesburg":  {"lat": -26.2041, "lon": 28.0473},
    "Houston":       {"lat": 29.7604,  "lon": -95.3698},
    "New York":      {"lat": 40.7128,  "lon": -74.0060},
    "Las Vegas":     {"lat": 36.1699,  "lon": -115.1398},
}

cities=["Madrid","Riyadh","Delhi","Jakarta","Sydney","São Paulo"]
lats=[city_dict[x]["lat"] for x in cities]
lons=[city_dict[x]["lon"] for x in cities]

#Run basic attribution for these cities
obs_cities=obs.sel(lat=lats,lon=lons,method="nearest")
coefs_cities=coefs_df.sel(lat=lats,lon=lons,method="nearest")

counts=obs_cities-dGMT*coefs_cities

#Generate assessments of beta relative uncertainty globally
relerrmap=100*coefs_df.std(dim="sample")/np.abs(coefs_df.mean(dim="sample"))

fs=7
cm=1/2.54
w1=5.5*cm
h1=4*cm
projection=ccrs.Robinson()
letters=list(string.ascii_lowercase)

plt.close()
widths=[w1]*3
heights=[h1*2]+[h1/12]+[h1*1.5]*2
fig=plt.figure(figsize=(sum(widths)+0.5,sum(heights)+0.5))
gs=fig.add_gridspec(ncols=len(widths),nrows=len(heights),width_ratios=widths,height_ratios=heights,wspace=0.3,hspace=0.3)

#Plot map of ERA5 uncertainty
ax3 = fig.add_subplot(gs[0,:], projection=projection)
#cax = fig.add_subplot(gs[1,:])
cax = inset_axes(ax3, width="50%", height="7%", loc="lower center",
                 bbox_to_anchor=(0, -0.08, 1, 1),
                 bbox_transform=ax3.transAxes, borderpad=0)

pl3=relerrmap.plot(ax=ax3,transform=ccrs.PlateCarree(),cmap="viridis",vmin=0,vmax=100,cbar_ax=cax,add_colorbar=True,cbar_kwargs={"orientation": "horizontal","label":r"Relative uncertainty, $\chi$ (%)","extend":"max"})
cbar = pl3.colorbar
cbar.set_label(r"Relative uncertainty, $\chi$ (%)", fontsize=fs)
cbar.ax.tick_params(labelsize=fs)

#cbar=pl3.colorbar
#cbar.set_label("Relative uncertainty (%)",fontsize=fs,rotation=0)
#cbar.ax.tick_params(labelsize=fs)
ax3.coastlines()
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.set_title("ERA5",fontsize=fs)
ax3.annotate(letters[0],xy=(-0.1,1.05),xycoords="axes fraction",fontsize=fs+1,fontweight="bold")
print("done global map")

for c, city in enumerate(cities):

	ax = fig.add_subplot(gs[2+int(c/3),c%3])
	
	ax.plot(obs_cities.year,obs_cities[:,c,c],c="tab:red",label="Observed",lw=1)
	ax.plot(counts.year,counts[:,c,c,:].median(dim="sample"),c="k",label="Counterfactual",lw=1)
	ax.fill_between(counts.year,counts[:,c,c,:].quantile(dim="sample",q=0.025),counts[:,c,c,:].quantile(dim="sample",q=0.975),color="k",alpha=0.4,lw=0)	

	if int(c/3)==1:
		ax.set_xlabel("Year",fontsize=fs)
	if c%3==0:
		ax.set_ylabel("T5d (°C)",fontsize=fs)
	ax.tick_params(labelsize=fs)	
	ax.set_title(city,fontsize=fs)
	ax.set_xlim([1940,2024])
	if c==0:
		ax.legend(fontsize=fs-1)
	ax.annotate(letters[1+c],xy=(-0.05,1.03),xycoords="axes fraction",fontsize=fs+1,fontweight="bold")
	print(city)

plt.savefig("figs/TX5_ERA5_example_95CI.png",dpi=300,bbox_inches="tight")
plt.close()


