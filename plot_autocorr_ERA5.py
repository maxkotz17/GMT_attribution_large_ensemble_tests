import xarray as xr
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import cartopy.crs as ccrs

#Signicance function on the autocorrelation 
def acf_significance(acf, T, alpha=0.05):
    """
    Compute significance bounds for autocorrelation function using Bartlett's formula.
    
    Parameters
    ----------
    acf : array of shape (nlags+1, n_series)
        Autocorrelation values from acf_vectorized
    T : int
        Length of original time series
    alpha : float
        Significance level (default 0.05 for 95% confidence)
    
    Returns
    -------
    stderr : array of shape (nlags+1, n_series)
        Standard errors for each lag and series
    conf_int : array of shape (nlags+1, n_series, 2)
        Confidence intervals (lower, upper) for each lag and series
    significant : array of shape (nlags+1, n_series)
        Boolean mask indicating which correlations are significant
    """
    from scipy import stats
    nlags_plus1, n_series = acf.shape
    nlags = nlags_plus1 - 1

    # Critical value for two-tailed test
    z_crit = stats.norm.ppf(1 - alpha/2)

    # Initialize standard errors
    stderr = np.zeros_like(acf)

    # Lag 0 always has stderr = 0 (correlation with self is always 1)
    stderr[0] = 0

    # Bartlett's formula for standard errors
    # SE(r_k) = sqrt((1 + 2 * sum(r_j^2 for j=1 to k-1)) / T)
    for k in range(1, nlags_plus1):
        if k == 1:
            # For lag 1, no previous lags to sum
            stderr[k] = 1.0 / np.sqrt(T)
        else:
            # Sum of squared autocorrelations up to lag k-1
            cumsum_sq = np.sum(acf[1:k]**2, axis=0)
            stderr[k] = np.sqrt((1 + 2 * cumsum_sq) / T)

    # Confidence intervals
    conf_int = np.zeros((nlags_plus1, n_series, 2))
    conf_int[:, :, 0] = acf - z_crit * stderr  # lower bound
    conf_int[:, :, 1] = acf + z_crit * stderr  # upper bound

    # Significance test: significant if doesn't contain 0
    significant = (conf_int[:, :, 0] > 0) | (conf_int[:, :, 1] < 0)

    return stderr, conf_int, significant

model="ERA5"
varn="tas"
metric="0.99_exp"
folder="/gpfs/scratch/bsc32/bsc400019/"

if "X" in metric:
	ACF=xr.open_dataset(folder + "attribution/autocorr/" + model + "_" + varn + "_" + metric + ".nc")
else:
	files=np.sort(glob.glob(folder + "attribution/autocorr/" + model + "_" + varn + "_" + metric + "*.nc"))
	ACF=[]
	for f, filen in enumerate(files):
		ACF.append(xr.open_dataset(filen))

#		if f==0:
#			ACF=xr.open_dataset(folder + "attribution/autocorr/" + model + "_" + varn + "_" + metric + "_" + str(f) + ".nc")
#		else:
#			x=xr.open_dataset(folder + "attribution/autocorr/" + model + "_" + varn + "_" + metric + "_" + str(f) + ".nc")
#			if model=="ERA5":
#				ACF=xr.concat((ACF,x),dim="lat")
#			else:
#				ACF=xr.concat((ACF,x),dim="member")

ACF=xr.concat(ACF,dim="lon")
ACF=ACF.acf.sortby("lon")
ACF=ACF.stack(z=("lat","lon"))

#Calculate significance of ACF
T=75 #Timeseries length, fixed because of using FFT to calculate autocorrelations
[stderr, conf_int, significance]=acf_significance(ACF,T)
significance=xr.DataArray(data=np.array(significance),dims=ACF.dims,coords={**ACF.coords},name="sig")

ACF=ACF.unstack("z")
significance=significance.unstack()

fs=6
cm=1/2.54
w1=8*cm
h1=5*cm

bounds=np.array([-1+x*0.2 for x in range(11)])
cmap=plt.get_cmap("RdBu",len(bounds)-1)
bnorm=mcolors.BoundaryNorm(boundaries=bounds,ncolors=len(bounds)-1)

plt.close()
widths=[w1]*5
heights=[h1]*4
fig=plt.figure(figsize=(sum(widths)+0.5,sum(heights)+1))
gs=fig.add_gridspec(ncols=len(widths),nrows=len(heights),width_ratios=widths,height_ratios=heights,wspace=0.3,hspace=0.4)

for l, lag in enumerate(ACF.lag.values[1:]):

	bounds=np.array([-1+x*0.2 for x in range(11)])
	cmap=plt.get_cmap("RdBu",len(bounds)-1)
	bnorm=mcolors.BoundaryNorm(boundaries=bounds,ncolors=len(bounds)-1)

	ax1 = fig.add_subplot(gs[int(l/5)*2,l%5],projection=ccrs.Robinson())

	pl1=ACF[lag,...].plot(ax=ax1,cmap=cmap,norm=bnorm,add_colorbar=False,transform=ccrs.PlateCarree())
	#significance[lag,...].astype(int).plot.contour(ax=ax1, levels=[0,0.5, 1.5],hatches=['','///'],colors='none',add_colorbar='False',transform=ccrs.PlateCarree(),)
	
	ax1.coastlines()
	ax1.set_title("Lag: " + str(lag),fontsize=fs)	
	cbar=fig.colorbar(pl1,ax=ax1,orientation='horizontal',pad=0.1,fraction=0.08)
	cbar.set_label('Autocorrelation',fontsize=fs)
	cbar.ax.tick_params(labelsize=fs)
	ax1.set_yticklabels([])
	ax1.set_xticklabels([])
	ax1.set_xlabel("")
	ax1.set_ylabel("")

	#Plot number of models with significant autocorrelation, according to bartlett's formulat
	cmap="Reds"	
	ax2 = fig.add_subplot(gs[int(l/5)*2+1,l%5],projection=ccrs.Robinson())
	pl2=significance.astype(int)[lag,...].plot(ax=ax2,cmap=cmap,vmin=0,vmax=1,add_colorbar=False,transform=ccrs.PlateCarree())
	ax2.set_title("")
	ax2.coastlines()
	cbar=fig.colorbar(pl2,ax=ax2,orientation='horizontal',pad=0.1,fraction=0.08)
	cbar.set_label('Significant',fontsize=fs)
	cbar.ax.tick_params(labelsize=fs)
	ax1.set_yticklabels([])
	ax1.set_xticklabels([])
	ax1.set_xlabel("")
	ax1.set_ylabel("")

plt.savefig('figs/autocor/AUTOCOR_' + varn + '_' + metric + '_' + model + '_sig.png',bbox_inches='tight',dpi=300)
plt.close()


