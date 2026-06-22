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

models=["MIROC6","MPI-ESM1-2-LR","CanESM5"]
varn=sys.argv[1]
metric=sys.argv[2] # 95/99
W=int(sys.argv[3])
relerr_cutoff=int(sys.argv[4])
ERA5=bool(int(sys.argv[5]))
endyr=sys.argv[6]

folder="/gpfs/scratch/bsc32/bsc400019/"

#Get uncertainty from true ensemble, bootstrap ensemble, their pattern correlations and then compare
if varn=="tas":
	relerrs=[(x+1)*10 for x in range(10)]
else:
	relerrs=[(x+1)*10 for x in range(20)]

area_frac=[]
area_frac_b=[]
corrws=[]

for m, model in enumerate(models):
	area_frac_b.append([])
	area_frac.append([])	

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
		coefs=xr.open_dataset(folder + "attribution/attr_coefs_" + model + "_" + varn + "_" + metric + "_" + endyr + ".nc")
		coefs_av=coefs.coef.mean(dim="member")
		coefs_std=coefs.coef.std(dim="member")

	coefs_relerr=100*coefs_std/np.abs(coefs_av)	
	#total surface-area (area weighted by latitude)
	areasum=float(coefs_relerr.where(coefs_relerr<0,np.cos(np.pi*coefs_relerr.lat/180)).sum())
	for r, relerr in enumerate(relerrs):
		#area where relative error is less than the threshold, summed
		area_frac[m].append(float(coefs_relerr.where(coefs_relerr>relerr,np.cos(np.pi*coefs_relerr.lat/180)).where(coefs_relerr<relerr,0).sum())/areasum)

	if "X" in metric:
		coefs_bs=xr.open_dataset(folder + "attribution/bootstrap/attr_coefs_" + model + "_" + varn + "_" + metric + "_block"+str(W)+"_" + endyr + ".nc")
	else:
		#load all the samples from the bootstrap
		files=np.sort(glob.glob(folder + "attribution/bootstrap/attr_coefs_" + model + "_" + varn + "_" + metric + "_block"+str(W)+"_seed*.nc"))
		for f, filen in enumerate(files):
			opn=xr.open_dataset(filen)
			if f==0:
				coefs_bs=opn
			else:
				coefs_bs=xr.concat((coefs_bs,opn),dim="sample")

	coefs_bs_av=coefs_bs.coef.mean(dim="sample")
	coefs_bs_std=coefs_bs.coef.std(dim="sample")
	coefs_bs_relerr=100*coefs_bs_std/np.abs(coefs_bs_av)

	for n in range(50):
		area_frac_b[m].append([])
		areasum=float(coefs_bs_relerr[n,...].where(coefs_bs_relerr[n,...]<0,np.cos(np.pi*coefs_bs_relerr.lat/180)).sum())
		for r, relerr in enumerate(relerrs):
			area_frac_b[m][n].append(float(coefs_bs_relerr[n,...].where(coefs_bs_relerr[n,...]>relerr,np.cos(np.pi*coefs_bs_relerr.lat/180)).where(coefs_bs_relerr[n,...]<relerr,0).sum())/areasum)
		#Pattern correlation - capping relerr at 100 because it can diverge where correlations =0
		coefs_bs_relerr[n,...]=coefs_bs_relerr[n,...].where(coefs_bs_relerr[n,...]<relerr_cutoff,relerr_cutoff)
		coefs_relerr=coefs_relerr.where(coefs_relerr<relerr_cutoff,relerr_cutoff)
		corrws.append(pat_corr_wght(coefs_relerr,coefs_bs_relerr[n,:,:]))

area_frac=np.array(area_frac)
area_frac_b=np.array(area_frac_b)
corrws=np.array(corrws)

df=pd.DataFrame()
modeln=[]
for x in range(3):
	modeln+=[models[x]]*50
df["model"]=modeln
df["Pattern cor"]=corrws

if ERA5==True:
	#Also get data for ERA5
	E5_area_frac=[]
	if "X" in metric:
		E5_coefs=xr.open_dataset(folder + "attribution/bootstrap/attr_coefs_ERA5_" + varn + "_" + metric.split("X5")[0] + "X5" + "_block" + str(W) + ".nc")
	else:
		files=glob.glob(folder + "attribution/bootstrap/attr_coefs_ERA5_" + varn + "_0.99_block"+str(W)+"*.nc")
		for f, filen in enumerate(files):
			opn=xr.open_dataset(filen)
			if f==0:
				E5_coefs=opn
			else:
				E5_coefs=xr.concat((E5_coefs,opn),dim="sample")

	E5_coefs_bs_av=E5_coefs.coef.mean(dim="sample")
	E5_coefs_bs_std=E5_coefs.coef.std(dim="sample")
	E5_coefs_bs_relerr=100*E5_coefs_bs_std/np.abs(E5_coefs_bs_av)

	areasum=float(E5_coefs_bs_relerr.where(E5_coefs_bs_relerr<0,np.cos(np.pi*E5_coefs_bs_relerr.lat/180)).sum())
	for r, relerr in enumerate(relerrs):
		E5_area_frac.append(float(E5_coefs_bs_relerr.where(E5_coefs_bs_relerr>relerr,np.cos(np.pi*E5_coefs_bs_relerr.lat/180)).where(E5_coefs_bs_relerr<relerr,0).sum())/areasum)

	E5_area_frac=np.array(E5_area_frac)

cols=["tab:blue","tab:orange","tab:green"]
fs=6
cm=1/2.54
w1=7*cm
h1=6*cm

plt.close()
widths=[w1]*2
heights=[h1]
fig=plt.figure(figsize=(sum(widths)+1,sum(heights)+0))
gs=fig.add_gridspec(ncols=len(widths),nrows=len(heights),width_ratios=widths,height_ratios=heights,wspace=0.25,hspace=0)

#projection = ccrs.Mollweide()

ax1 = fig.add_subplot(gs[0])#, projection=projection)
for m, model in enumerate(models):
	ax1.plot(relerrs,area_frac[m,:],label=model + ": ensemble",c=cols[m])
	ax1.plot(relerrs,np.median(area_frac_b[m,...],axis=0),label=model + ": bootstrap",c=cols[m],ls='--')	
	ax1.fill_between(relerrs,np.percentile(area_frac_b[m,...],q=5,axis=0),np.percentile(area_frac_b[m,...],q=95,axis=0),color=cols[m],alpha=0.15)

if ERA5==True:
	ax1.plot(relerrs,E5_area_frac,label="ERA5: bootstrap",c="k",ls="--")

ax1.set_xlabel("Relative error due to internal variability (%)",fontsize=fs)
ax1.set_ylabel("Cumulative surface area fraction",fontsize=fs)
ax1.legend(fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs)
#ax1.axvline(x=40,ls="--",c="k")
ax1.grid(which="major",axis="both")
ax1.set_ylim([0,1])
if varn=="tas":
	ax1.set_xlim([10,100])
else:
	ax1.set_xlim([10,200])
ax1.annotate("a",xy=(-0.1,1.05),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

ax1=fig.add_subplot(gs[1])
ax1.grid(which="major",axis="both")
sns.violinplot(data=df,x="model",y="Pattern cor",ax=ax1,palette=cols)
ax1.set_ylim([0,1])
ax1.set_ylabel("Weighted pattern correlation between\nlarge ensemble and bootstrapped members",fontsize=fs)
ax1.set_xlabel("Model",fontsize=fs)
ax1.tick_params(axis='both', which='major', labelsize=fs)
ax1.annotate("b",xy=(-0.1,1.05),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

if ERA5==True:
	plt.savefig('figs/member_stats/' + varn + '_' + metric + '_uncertainty_ag_bootens_block' + str(W) + '_' + str(relerr_cutoff) + '_ERA5.png',bbox_inches='tight',dpi=300)
else:
	plt.savefig('figs/member_stats/' + varn + '_' + metric + '_uncertainty_ag_bootens_block' + str(W) + '_' + str(relerr_cutoff) + '_' + endyr + '.png',bbox_inches='tight',dpi=300)
plt.close()

print(area_frac[:,3])
print(models)
print(df.groupby("model").median())
