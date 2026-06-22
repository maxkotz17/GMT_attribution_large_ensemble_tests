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

model=sys.argv[1] #'GFDL-ESM4'
varn=sys.argv[2]
metric=sys.argv[3] # 95/99
W=int(sys.argv[4])
folder="/gpfs/scratch/bsc32/bsc400019/"
#Load data
if "X" in metric:
        coefs=xr.open_dataset(folder + "attribution/attr_coefs_" + model + "_" + varn + "_" + metric + ".nc")
        coefs_av=coefs.coef.mean(dim="member")
        coefs_std=coefs.coef.std(dim="member")
else:
	coefs=[]
	#range of coefficients across the true ensemble
	for i in range(50):
		coefs.append(xr.open_dataset(folder + "attribution/attr_coefs_" + model + "_" + varn + "_" + metric + "_" + str(i) + ".nc"))
	coefs=xr.concat((coefs),dim="member")
	coefs=coefs.sortby("lon")
	#downsample according to what we did for the "observational ensemble"
	coefs_av=coefs.coef[:,:,:].mean(dim="member")
	coefs_std=coefs.coef[:,:,:].std(dim="member")
coefs_relerr=100*coefs_std/np.abs(coefs_av)
coefs_SN=np.abs(coefs_av)/coefs_std
#for the sake of plotting and correlation coefficient calculation, set those regions with relative error > 100 as 100
coefs_relerr=coefs_relerr.where(coefs_relerr<100,100)

#Bootstrap results
if "X" in metric:
        coefs_bs=xr.open_dataset(folder + "attribution/bootstrap/attr_coefs_" + model + "_" + varn + "_" + metric + "_block"+str(W)+".nc")
else:
        #load all the samples from the bootstrap
        files=np.sort(glob.glob(folder + "attribution/bootstrap/attr_coefs_" + model + "_" + varn + "_" + metric + "_block"+str(W)+"_seed*.nc"))
        for f, filen in enumerate(files):
                opn=xr.open_dataset(filen)
                if f==0:
                        coefs_bs=opn
                else:
                        coefs_bs=xr.concat((coefs_bs,opn),dim="sample")

coefs_bs_av=coefs_bs["coef"].mean(dim="sample")
coefs_bs_std=coefs_bs["coef"].std(dim="sample")
coefs_bs_relerr=100*coefs_bs_std/np.abs(coefs_bs_av)
coefs_SN=np.abs(coefs_av)/coefs_std
coefs_bs_relerr=coefs_bs_relerr.where(coefs_bs_relerr<100,100)

fs=7
cm=1/2.54
w1=7*cm
h1=5*cm

if varn=="tas":
	vmax=3
elif varn=="pr":
	vmax=1.5
plt.close()
widths=[w1]*5
heights=[h1]*3
fig=plt.figure(figsize=(sum(widths)+0.2,sum(heights)+1))
gs=fig.add_gridspec(ncols=len(widths),nrows=len(heights),width_ratios=widths,height_ratios=heights,wspace=0.2,hspace=0.4)

projection = ccrs.Robinson()

ax1 = fig.add_subplot(gs[0,0], projection=projection)
pl1=coefs_av.plot(ax=ax1,cmap="viridis",vmin=0,vmax=vmax,add_colorbar=False,transform=ccrs.PlateCarree())
ax1.coastlines()
ax1.set_title("Large ensemble",fontsize=fs)
cbar=fig.colorbar(pl1,ax=ax1,orientation='horizontal',pad=0.1,fraction=0.08)
cbar.set_label('Average correlation with GMT\n(logarithmic change per K)',fontsize=fs)
cbar.ax.tick_params(labelsize=fs)
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_xlabel("")
ax1.set_ylabel("")

ax2 = fig.add_subplot(gs[1,0], projection=projection)
pl2=coefs_std.plot(ax=ax2,cmap="viridis",vmin=0,vmax=vmax,add_colorbar=False,transform=ccrs.PlateCarree())
ax2.coastlines()
ax2.set_title('')
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_xlabel("")
ax2.set_ylabel("")
cbar=fig.colorbar(pl2,ax=ax2,orientation='horizontal',pad=0.1,fraction=0.08)
cbar.set_label('STD of correlations with GMT\n(logarithmic change per K)',fontsize=fs)
cbar.ax.tick_params(labelsize=fs)

ax3 = fig.add_subplot(gs[2,0], projection=projection)
pl3=coefs_relerr.plot(ax=ax3,cmap="viridis",vmin=0,vmax=100,add_colorbar=False,transform=ccrs.PlateCarree())
ax3.coastlines()
ax3.set_xticklabels([])
ax3.set_yticklabels([])
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.set_title('')
cbar=fig.colorbar(pl3,ax=ax3,orientation='horizontal',pad=0.1,fraction=0.08)
cbar.set_label('Relative uncertainty (%)',fontsize=fs)
cbar.ax.tick_params(labelsize=fs)

fig.suptitle(model + " " + varn + " " + metric,fontsize=fs)

#plot results from observational ensemble
for m, member in enumerate(coefs_bs_av.member[:4].values):	
	corr=np.round(float(xr.corr(coefs_av,coefs_bs_av[m,...],dim=["lat","lon"])),decimals=2)
	corrw=np.round(pat_corr_wght(coefs_av,coefs_bs_av[m,...]),decimals=2)
	ax1=fig.add_subplot(gs[0,1+m],projection=projection)	
	pl1=coefs_bs_av[m,...].plot(ax=ax1,cmap="viridis",vmin=0,vmax=vmax,add_colorbar=False,transform=ccrs.PlateCarree())
	ax1.coastlines()
	ax1.set_title("Member:" + member + ": " + str(corrw),fontsize=fs)
	cbar=fig.colorbar(pl1,ax=ax1,orientation='horizontal',pad=0.1,fraction=0.08)
	cbar.set_label('Average correlation with GMT\n(logarithmic change per K)',fontsize=fs)
	cbar.ax.tick_params(labelsize=fs)
	ax1.set_yticklabels([])
	ax1.set_xticklabels([])
	ax1.set_xlabel("")
	ax1.set_ylabel("")

	corr=np.round(float(xr.corr(coefs_std,coefs_bs_std[m,...],dim=["lat","lon"])),decimals=2)
	corrw=np.round(pat_corr_wght(coefs_std,coefs_bs_std[m,...]),decimals=2)
	ax2=fig.add_subplot(gs[1,1+m],projection=projection)
	pl2=coefs_bs_std[m,...].plot(ax=ax2,cmap="viridis",vmin=0,vmax=vmax,add_colorbar=False,transform=ccrs.PlateCarree())
	ax2.coastlines()
	ax2.set_title(str(corrw),fontsize=fs)
	ax2.set_yticklabels([])
	ax2.set_xticklabels([])
	cbar=fig.colorbar(pl2,ax=ax2,orientation='horizontal',pad=0.1,fraction=0.08)
	cbar.set_label('STD of correlations with GMT\n(logarithmic change per K)',fontsize=fs)
	cbar.ax.tick_params(labelsize=fs)
	ax2.set_ylabel("")
	ax2.set_xlabel("")

	corr=np.round(float(xr.corr(coefs_relerr,coefs_bs_relerr[m,...],dim=["lat","lon"])),decimals=2)
	corrw=np.round(pat_corr_wght(coefs_relerr,coefs_bs_relerr[m,...]),decimals=2)
	ax3=fig.add_subplot(gs[2,1+m],projection=projection)
	coefs_bs_relerr[m,...].plot(ax=ax3,cmap="viridis",vmin=0,vmax=100,add_colorbar=False,transform=ccrs.PlateCarree())
	ax3.coastlines()
	ax3.set_title(str(corrw),fontsize=fs)
	ax3.set_xticklabels([])
	ax3.set_yticklabels([])	
	ax3.set_xlabel("")
	ax3.set_ylabel("")
	cbar=fig.colorbar(pl3,ax=ax3,orientation='horizontal',pad=0.1,fraction=0.08)
	cbar.set_label('Relative uncertainty (%)',fontsize=fs)
	cbar.ax.tick_params(labelsize=fs)


plt.savefig('figs/boot_stats/' + varn + '_' + metric + '_' + model + '_' + str(W) + '.png',bbox_inches='tight',dpi=300)
plt.close()

