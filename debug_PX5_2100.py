import xarray as xr
import numpy as np
import sys
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

varn="pr"
metric="X5"
models=["MPI-ESM1-2-LR","MIROC6","CanESM5"]
endyr=2100

fs=7
cm=1/2.54
w1=7.5*cm
h1=4*cm
projection=ccrs.Robinson()
import string
letters=list(string.ascii_lowercase)

plt.close()
widths=[w1]*3
heights=[h1]*5
fig=plt.figure(figsize=(sum(widths)+0.3,sum(heights)+0.3*4))
gs=fig.add_gridspec(ncols=len(widths),nrows=len(heights),width_ratios=widths,height_ratios=heights,wspace=0.1,hspace=0.3)

#Output the standard deviation across models of the % PX5 by end of century
for m, model in enumerate(models):

	#load data
	climex=xr.open_dataset("clim_extremes/" + varn + "/" + model + "_" + varn + "_" + metric + ".nc")
	if climex.lon.max().values>180:
		climex['lon'] = (climex['lon'].values + 180) % 360 - 180
		climex=climex.sortby("lon")
	climex=climex.to_array(name=varn+"_exp").squeeze()
	#Match exactly ERA5 time-frame 
	climex=climex[:,climex.year.values>1939,:,:]
	climex=climex.loc[:,climex.year.values<endyr,...]
	#If heavy precip extreme intensity express as % change of baseline, for consistency with clausius clapeyron
	#if (metric=="X5"):
	#	climex=100*climex/(climex.loc[:,climex.year<1980,:,:].mean(dim="year"))

	print(model)
	print(climex.loc[:,climex.year>2080,:,:].mean(dim="year").std(dim="member").mean(dim=["lat","lon"]))	
	print(climex.loc[:,climex.year.isin([x for x in range(2000,2020)]),:,:].mean(dim="year").std(dim="member").mean(dim=["lat","lon"]))

	base=(climex.loc[:,climex.year<1980,:,:].mean(dim="year"))
	diff=climex.loc[:,climex.year>2080,:,:].mean(dim=["year"])-base
	pdiff=(climex.loc[:,climex.year>2080,:,:].mean(dim=["year"])-base)/base

        ax3 = fig.add_subplot(gs[0,m], projection=projection) 
	cax = inset_axes(ax3, width="50%", height="5%", loc="lower center",
			 bbox_to_anchor=(0, -0.08, 1, 1),
			 bbox_transform=ax3.transAxes, borderpad=0)
 
        pl3=diff.mean(dim="member").plot(ax=ax3,transform=ccrs.PlateCarree(),cmap="RdBu",vmin=-20,vmax=20,cbar_ax=cax,add_colorbar=True,cbar_kwargs={"orientation": "horizontal"})
        cbar=pl3.colorbar
        cbar.ax.tick_params(labelsize=fs)
        ax3.coastlines()
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_xlabel("")
        ax3.set_ylabel("")
        ax3.set_title('')
	
        ax3 = fig.add_subplot(gs[1,m], projection=projection) 
        cax = inset_axes(ax3, width="50%", height="5%", loc="lower center",
                         bbox_to_anchor=(0, -0.08, 1, 1),
                         bbox_transform=ax3.transAxes, borderpad=0)
 
        pl3=diff.std(dim="member").plot(ax=ax3,transform=ccrs.PlateCarree(),cmap="viridis",cbar_ax=cax,add_colorbar=True,cbar_kwargs={"orientation": "horizontal"})
        cbar=pl3.colorbar
        cbar.ax.tick_params(labelsize=fs)
        ax3.coastlines()
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_xlabel("")
        ax3.set_ylabel("")
        ax3.set_title('')

        ax3 = fig.add_subplot(gs[2,m], projection=projection) 
        cax = inset_axes(ax3, width="50%", height="5%", loc="lower center",
                         bbox_to_anchor=(0, -0.08, 1, 1),
                         bbox_transform=ax3.transAxes, borderpad=0)

        pl3=pdiff.mean(dim="member").plot(ax=ax3,transform=ccrs.PlateCarree(),cmap="RdBu",vmin=-3,vmax=3,cbar_ax=cax,add_colorbar=True,cbar_kwargs={"orientation": "horizontal"})
        cbar=pl3.colorbar
        cbar.ax.tick_params(labelsize=fs)
        ax3.coastlines()
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_xlabel("")
        ax3.set_ylabel("")
        ax3.set_title('')

        ax3 = fig.add_subplot(gs[3,m], projection=projection) 
        cax = inset_axes(ax3, width="50%", height="5%", loc="lower center",
                         bbox_to_anchor=(0, -0.08, 1, 1),
                         bbox_transform=ax3.transAxes, borderpad=0)
 
        pl3=pdiff.std(dim="member").plot(ax=ax3,transform=ccrs.PlateCarree(),cmap="viridis",vmin=0,vmax=0.5,cbar_ax=cax,add_colorbar=True,cbar_kwargs={"orientation": "horizontal"})
        cbar=pl3.colorbar
        cbar.ax.tick_params(labelsize=fs)
        ax3.coastlines()
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_xlabel("")
        ax3.set_ylabel("")
        ax3.set_title('')

        ax3 = fig.add_subplot(gs[4,m], projection=projection)
        cax = inset_axes(ax3, width="50%", height="5%", loc="lower center",
                         bbox_to_anchor=(0, -0.08, 1, 1),
                         bbox_transform=ax3.transAxes, borderpad=0)

        pl3=(pdiff.std(dim="member")/abs(pdiff.mean(dim="member"))).plot(ax=ax3,transform=ccrs.PlateCarree(),cmap="viridis",vmin=0,vmax=2,cbar_ax=cax,add_colorbar=True,cbar_kwargs={"orientation": "horizontal"})
        cbar=pl3.colorbar
        cbar.ax.tick_params(labelsize=fs)
        ax3.coastlines()
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])
        ax3.set_xlabel("")
        ax3.set_ylabel("")
        ax3.set_title('')

plt.savefig("PX5_debug.png",dpi=300,bbox_inches="tight")
plt.close()

for m, model in enumerate(models):

        #load data
        climex=xr.open_dataset("clim_extremes/" + varn + "/" + model + "_" + varn + "_" + metric + ".nc")
        if climex.lon.max().values>180:
                climex['lon'] = (climex['lon'].values + 180) % 360 - 180
                climex=climex.sortby("lon")
        climex=climex.to_array(name=varn+"_exp").squeeze()
        #Match exactly ERA5 time-frame 
        climex=climex[:,climex.year.values>1939,:,:]
        climex=climex.loc[:,climex.year.values<endyr,...]
        #If heavy precip extreme intensity express as % change of baseline, for consistency with clausius clapeyron
        #if (metric=="X5"):
        #       climex=100*climex/(climex.loc[:,climex.year<1980,:,:].mean(dim="year"))

        #print(model)
        #print(climex.loc[:,climex.year>2080,:,:].mean(dim="year").std(dim="member").mean(dim=["lat","lon"]))
        #print(climex.loc[:,climex.year.isin([x for x in range(2000,2020)]),:,:].mean(dim="year").std(dim="member").mean(dim=["lat","lon"]))

        base=(climex.loc[:,climex.year<1980,:,:].mean(dim="year"))
        diff=climex.loc[:,climex.year>2080,:,:].mean(dim=["year"])-base
        pdiff=(climex.loc[:,climex.year>2080,:,:].mean(dim=["year"])-base)/base

	relerr_pdiff=pdiff.std(dim="member")/abs(pdiff.mean(dim="member"))

	folder="/gpfs/scratch/bsc32/bsc400019/"
	coefs_df=xr.open_dataset(folder + "attribution/attr_coefs_" + model + "_" + varn + "_" + metric + "_" + str(endyr) + ".nc")

	relerr_coefs=coefs_df.coef.std(dim="member")/abs(coefs_df.coef.mean(dim="member"))

	relerr_pdiff=pdiff.std(dim="member")/abs(pdiff.mean(dim="member"))

	folder="/gpfs/scratch/bsc32/bsc400019/"
	coefs_df=xr.open_dataset(folder + "attribution/attr_coefs_" + model + "_" + varn + "_" + metric + "_" + str(endyr) + ".nc")

	relerr_coefs=coefs_df.coef.std(dim="member")/abs(coefs_df.coef.mean(dim="member"))

	print(model)
	print((relerr_pdiff<1).sum()/(len(climex.lat)*len(climex.lon)))
	print((relerr_coefs<1).sum()/(len(climex.lat)*len(climex.lon)))



