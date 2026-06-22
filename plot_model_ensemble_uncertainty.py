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

model=sys.argv[1] #'GFDL-ESM4'
varn=sys.argv[2]
metric=sys.argv[3] # 95/99
W=sys.argv[4]

if model=="ERA5":	
	if "X" in metric:
		coefs=xr.open_dataset("attribution/bootstrap/attr_coefs_" + model + "_" + varn + "_" + metric + "_block" + str(W) + ".nc")
	else:
		files=glob.glob("attribution/bootstrap/attr_coefs_*" + model + "_" + varn + "_" + metric + "_block" + str(W) + "*.nc")
		for f, filen in enumerate(files):
			x=xr.open_dataset(filen)
			if f==0:
				coefs=x
			else:
				coefs=xr.concat((coefs,x),dim="sample")
	coefs_av=coefs["coef"].mean(dim="sample")
	coefs_std=coefs["coef"].std(dim="sample")
	coefs_relerr=100*coefs_std/np.abs(coefs_av)
else:
	coefs=xr.open_dataset("attribution/attr_coefs_" + model + "_" + varn + "_" + thresh + "_rollingcentre.nc")
	coefs_av=coefs[varn + "_exp"].mean(dim="member")
	coefs_std=coefs[varn + "_exp"].std(dim="member")
	coefs_relerr=100*coefs_std/np.abs(coefs_av)

fs=6
cm=1/2.54
w1=8*cm
h1=5*cm

plt.close()
widths=[w1]
heights=[h1]*3
fig=plt.figure(figsize=(sum(widths)+0.5,sum(heights)+1))
gs=fig.add_gridspec(ncols=len(widths),nrows=len(heights),width_ratios=widths,height_ratios=heights,wspace=0.5,hspace=0.5)

#projection = ccrs.Mollweide()

ax1 = fig.add_subplot(gs[0])#, projection=projection)
#pl1=dataset.FREQ_EXC.plot(ax=ax1,transform=ccrs.PlateCarree(),cmap='viridis',vmin=0,vmax=80,add_colorbar=False)
pl1=coefs_av.plot(ax=ax1,cmap="viridis",vmin=0,vmax=3,add_colorbar=False)
#ax1.coastlines()
ax1.set_title(model + " " + varn + ">" + metric,fontsize=fs)
cbar=fig.colorbar(pl1,ax=ax1,orientation='horizontal',pad=0.1,fraction=0.08)
cbar.set_label('Average correlation with GMT\n(logarithmic change per K)',fontsize=fs)
cbar.ax.tick_params(labelsize=fs)
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_xlabel("")
ax1.set_ylabel("")

ax2 = fig.add_subplot(gs[1])#, projection=projection)
pl2=coefs_std.plot(ax=ax2,cmap="viridis",vmin=0,vmax=3,add_colorbar=False)
#ax2.coastlines()
ax2.set_title('')
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_xlabel("")
ax2.set_ylabel("")
cbar=fig.colorbar(pl2,ax=ax2,orientation='horizontal',pad=0.1,fraction=0.08)
cbar.set_label('STD of correlations with GMT\n(logarithmic change per K)',fontsize=fs)
cbar.ax.tick_params(labelsize=fs)

vmin=-10
vmax=100
vcenter=1
norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
ax3 = fig.add_subplot(gs[2])#, projection=projection)
pl3=coefs_relerr.plot(ax=ax3,cmap="viridis",vmin=0,vmax=100,add_colorbar=False)
ax3.set_yticklabels([])
ax3.set_xticklabels([])
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.set_title('')
cbar=fig.colorbar(pl3,ax=ax3,orientation='horizontal',pad=0.1,fraction=0.08)
cbar.set_label('Relative uncertainty (%)',fontsize=fs)
cbar.ax.tick_params(labelsize=fs)

fig.suptitle('Statistics of GMT-correlation across 51 ensemble members',fontsize=fs)

plt.savefig('figs/member_stats/' + varn + '_' + metric + '_' + model + '_rollingcentre.png',bbox_inches='tight',dpi=300)
plt.close()

