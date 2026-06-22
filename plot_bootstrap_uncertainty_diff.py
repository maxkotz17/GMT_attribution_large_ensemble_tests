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
thresh=sys.argv[3] # 95/99

#range of coefficients across the true ensemble
coefs=xr.open_dataset("attribution/attr_coefs_" + model + "_" + varn + "_" + thresh + "_rollingcentre.nc")
#downsample according to what we did for the "observational ensemble"
coefs_av=coefs[varn + "_exp"][:,:,:].mean(dim="member")
coefs_std=coefs[varn + "_exp"][:,:,:].std(dim="member")
coefs_relerr=100*coefs_std/np.abs(coefs_av)
coefs_SN=np.abs(coefs_av)/coefs_std
#for the sake of plotting and correlation coefficient calculation, set those regions with relative error > 100 as 100
coefs_relerr=coefs_relerr.where(coefs_relerr<100,100)

W=2
#load all the samples from the bootstrap
files=np.sort(glob.glob("attribution/bootstrap/attr_coefs_" + model + "*block"+str(W)+"_seed*.nc"))
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

coefs_relerr_diff=100*(coefs_bs_relerr-coefs_relerr)/coefs_relerr

cmax=50
fs=6
cm=1/2.54
w1=8*cm
h1=5*cm

plt.close()
widths=[w1]
heights=[h1]*2
fig=plt.figure(figsize=(sum(widths)+0.5,sum(heights)+1))
gs=fig.add_gridspec(ncols=len(widths),nrows=len(heights),width_ratios=widths,height_ratios=heights,wspace=0.5,hspace=0.5)

norm = mcolors.TwoSlopeNorm(vmin=-cmax, vcenter=0, vmax=cmax)
ax1 = fig.add_subplot(gs[0,0])#, projection=projection)
pl1=coefs_relerr_diff.mean(dim="member").plot(ax=ax1,cmap="RdBu",norm=norm,add_colorbar=False)
#ax1.coastlines()
ax1.set_title("Average difference in rel. error",fontsize=fs)
cbar=fig.colorbar(pl1,ax=ax1,orientation='horizontal',pad=0.1,fraction=0.08)
cbar.set_label('Average difference in rel. error',fontsize=fs)
cbar.ax.tick_params(labelsize=fs)
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_xlabel("")
ax1.set_ylabel("")

#hatch based on p-test
plt.rcParams['hatch.linewidth'] = 0.2
#Condition where there is a >2/3 chance of the bootstrapped error being smaller
condition=(coefs_relerr_diff<0).sum(dim="member")>33
#Hatch where condition is true
masked = np.ma.masked_where(~condition, condition)
ax1.contourf(
    condition["lon"], condition["lat"], condition,
    levels=[0.5, 1],  # only fill where condition == True
    hatches=["///////"], colors="none",edgecolor="k"
)

ax2 = fig.add_subplot(gs[1])
pl2=coefs_relerr_diff.std(dim="member").plot(ax=ax2,cmap="Reds",vmin=0,vmax=cmax,add_colorbar=False)
ax2.set_title("Standard dev. of difference in rel.error",fontsize=fs)
cbar=fig.colorbar(pl2,ax=ax2,orientation='horizontal',pad=0.1,fraction=0.08)
cbar.set_label('Standard dev. of difference in rel.error',fontsize=fs)
cbar.ax.tick_params(labelsize=fs)
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_xlabel("")
ax2.set_ylabel("")

plt.savefig('figs/boot_stats/' + varn + '_' + thresh + '_' + model + '_difference.png',bbox_inches='tight',dpi=300)
plt.close()


