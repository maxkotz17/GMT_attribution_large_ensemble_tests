import numpy as np
import pandas as pd
import xarray as xr
#import cartopy.crs as ccrs
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

def pat_corr_wght(da1, da2, lat_name="lat", lon_name="lon"):
    """
    Calculate area-weighted pattern correlation between two DataArrays.

    Parameters
    ----------
    da1, da2 : xr.DataArray
        Input data arrays with dimensions including lat/lon.
    lat_name : str
        Name of latitude dimension (default "lat").
    lon_name : str
        Name of longitude dimension (default "lon").

    Returns
    -------
    corr : float
        Area-weighted Pearson correlation coefficient.
    """
    # Align the two DataArrays
    da1, da2 = xr.align(da1, da2)

    # Compute weights: proportional to cos(lat)
    weights = np.cos(np.deg2rad(da1[lat_name]))
    weights = weights / weights.mean()  # normalize (optional)
    
    # Apply weights along latitude
    # Need to broadcast weights to full 2D shape
    w2d = weights.broadcast_like(da1)

    # Flatten, mask NaNs
    x = da1.values.flatten()
    y = da2.values.flatten()
    w = w2d.values.flatten()

    mask = np.isfinite(x) & np.isfinite(y)
    x, y, w = x[mask], y[mask], w[mask]

    # Weighted means
    x_mean = np.average(x, weights=w)
    y_mean = np.average(y, weights=w)

    # Weighted covariance and variances
    cov = np.average((x - x_mean) * (y - y_mean), weights=w)
    var_x = np.average((x - x_mean) ** 2, weights=w)
    var_y = np.average((y - y_mean) ** 2, weights=w)

    return cov / np.sqrt(var_x * var_y)

models=["MPI-ESM1-2-LR","MIROC6","CanESM5"]
varn=sys.argv[1]
metric=sys.argv[2]
W=int(sys.argv[3])

folder="/gpfs/scratch/bsc32/bsc400019/"

if varn=="tas":
	vmax=100
elif varn=="pr":
	vmax=200

fs=7
cm=1/2.54
w1=7.5*cm
h1=4*cm
projection=ccrs.Robinson()
letters=list(string.ascii_lowercase)

plt.close()
widths=[w1]*4+[w1/12]
heights=[h1]*3
fig=plt.figure(figsize=(sum(widths)+0.3,sum(heights)+0.4*2))
gs=fig.add_gridspec(ncols=len(widths),nrows=len(heights),width_ratios=widths,height_ratios=heights,wspace=0.1,hspace=0.4)

for m, model in enumerate(models+["ERA5"]):

	if model=="ERA5":
		method="bootstrap"
	else:
		method="large ensemble"
	if model!="ERA5":
		if "0.99" in metric:	
			coefs=[]
			for i in range(50):
				coefs.append(xr.open_dataset(folder + "attribution/attr_coefs_" + model + "_" + varn + "_" + metric + "_" + str(i) + ".nc"))
			coefs=xr.concat(coefs,dim="member")
			coefs=coefs.sortby("lon")
			#range of coefficients across the true ensemble
			#coefs=xr.open_dataset("attribution/attr_coefs_" + model + "_" + varn + "_" + metric + "_rollingcentre.nc")
			#downsample according to what we did for the "observational ensemble"
			coefs_av=coefs.coef[:,:,:].mean(dim="member")
			coefs_std=coefs.coef[:,:,:].std(dim="member")
		else:
			coefs=xr.open_dataset(folder + "attribution/attr_coefs_" + model + "_" + varn + "_" + metric + ".nc")
			coefs_av=coefs.coef.mean(dim="member")
			coefs_std=coefs.coef.std(dim="member")
	else:
		coefs=xr.open_dataset(folder + "attribution/bootstrap/attr_coefs_ERA5_" + varn + "_" + metric + "_block" + str(W) + ".nc").coef
		coefs_av=coefs.mean(dim="sample")
		coefs_std=coefs.std(dim="sample")

	coefs_relerr=100*coefs_std/np.abs(coefs_av)
	coefs_SN=np.abs(coefs_av)/coefs_std
	#for the sake of plotting and correlation coefficient calculation, set those regions with relative error > 100 as 100
	coefs_relerr=coefs_relerr.where(coefs_relerr<vmax,vmax)

	#plot the true ensemble relative uncertainty
	ax3 = fig.add_subplot(gs[0,m], projection=projection)
	if m==0:
		cax= fig.add_subplot(gs[0,4])
		pl3=coefs_av.plot(ax=ax3,transform=ccrs.PlateCarree(),cmap="RdBu_r",vmin=-2,vmax=2,cbar_ax=cax,add_colorbar=True,cbar_kwargs={"orientation": "vertical","label":"Relative uncertainty (%)","extend":"both"})
		cbar=pl3.colorbar
		cbar.set_label("Mean GMST-T5d slope\n(°C K$^{-1}$)",fontsize=fs,rotation=90)
		cbar.ax.tick_params(labelsize=fs)
	else:
		pl3=coefs_av.plot(ax=ax3,transform=ccrs.PlateCarree(),cmap="RdBu_r",vmin=-2,vmax=2,add_colorbar=False)
	ax3.coastlines()
	ax3.set_xticklabels([])
	ax3.set_yticklabels([])
	ax3.set_xlabel("")
	ax3.set_ylabel("")
	ax3.set_title('')
	ax3.set_title(model + ": " + method,fontsize=fs)
	ax3.annotate(letters[m*3],xy=(-0.1,1.05),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

	ax3 = fig.add_subplot(gs[1,m], projection=projection)
	if m==0:
		cax= fig.add_subplot(gs[1,4])
		pl3=coefs_std.plot(ax=ax3,transform=ccrs.PlateCarree(),cmap="viridis",vmin=0,vmax=2,cbar_ax=cax,add_colorbar=True,cbar_kwargs={"orientation": "vertical","label":"Relative uncertainty (%)","extend":"max"})
		cbar=pl3.colorbar
		cbar.set_label("Absolute uncertainty,\n"+r"$\sigma$ (°C K$^{-1}$)",fontsize=fs,rotation=90)
		cbar.ax.tick_params(labelsize=fs)
	else:
		pl3=coefs_std.plot(ax=ax3,transform=ccrs.PlateCarree(),cmap="viridis",vmin=0,vmax=2,add_colorbar=False)
	ax3.coastlines()
	ax3.set_xticklabels([])
	ax3.set_yticklabels([])
	ax3.set_xlabel("")
	ax3.set_ylabel("")
	ax3.set_title('')
	ax3.set_title(model + ": " + method,fontsize=fs)
	ax3.annotate(letters[m*3+1],xy=(-0.1,1.05),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

	ax3 = fig.add_subplot(gs[2,m], projection=projection)
	if m==0:
		cax= fig.add_subplot(gs[2,4])
		pl3=coefs_relerr.plot(ax=ax3,transform=ccrs.PlateCarree(),cmap="viridis",vmin=0,vmax=vmax,cbar_ax=cax,add_colorbar=True,cbar_kwargs={"orientation": "vertical","label":"Relative uncertainty (%)","extend":"max"})
		cbar=pl3.colorbar
		cbar.set_label("Relative uncertainty,\n"+r"$\chi$ (%)",fontsize=fs,rotation=90)
		cbar.ax.tick_params(labelsize=fs)
	else:
		pl3=coefs_relerr.plot(ax=ax3,transform=ccrs.PlateCarree(),cmap="viridis",vmin=0,vmax=vmax,add_colorbar=False)
	ax3.coastlines()
	ax3.set_xticklabels([])
	ax3.set_yticklabels([])
	ax3.set_xlabel("")
	ax3.set_ylabel("")
	ax3.set_title('')
	ax3.set_title(model + ": " + method,fontsize=fs)
	
	ax3.annotate(letters[m*3+2],xy=(-0.1,1.05),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")
	print("Done" + model)

plt.savefig('figs/boot_stats/UNCERTAINTY_BREAKDOWN_' + varn + '_' + metric + '_block' + str(W) + '_summary.png',bbox_inches='tight',dpi=300)
plt.close()

