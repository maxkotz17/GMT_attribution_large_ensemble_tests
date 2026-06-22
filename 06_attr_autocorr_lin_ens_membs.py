import xarray as xr
import numpy as np
import pandas as pd
import scipy.interpolate
import sys
import statsmodels.api as sm
import statsmodels.formula.api as smf
#plot a few examples
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from matplotlib import gridspec
import seaborn as sns

#funnction for RMSE of two arrays of arbitrary dimensions
def RMSE(arr1,arr2,axis):
    diff=arr1-arr2
    return(np.sqrt(np.mean(diff**2,axis=axis)))

#function for pearson correlation of two arrays of arbitrary dimensions
def pearson_corr(arr1,arr2,axis):
    mean1=np.mean(arr1,axis=axis,keepdims=True)
    mean2=np.mean(arr2,axis=axis,keepdims=True)
    std1=np.std(arr1,axis=axis)
    std2=np.std(arr2,axis=axis)
    cov=np.mean((arr1-mean1)*(arr2-mean2),axis=axis)
    return(cov/(std1*std2))

def lin_reg_2D(X,Y):
	# Add intercept column
	X_mat = np.column_stack([np.ones(len(X)), X])  # shape (n_samples, 2)

	# Solve regression
	beta, residuals, rank, s = np.linalg.lstsq(X_mat, Y, rcond=None)
	# beta shape: (2, n_targets)
	intercept = beta[0, :]  # first row = intercepts
	slope = beta[1, :]      # second row = slopes

	# Predicted values and residuals
	Y_hat = X_mat @ beta
	resids = Y - Y_hat

	# Explained variance (R^2)
	SS_tot = ((Y - Y.mean(axis=0))**2).sum(axis=0)
	SS_res = ((Y - Y_hat)**2).sum(axis=0)
	R2 = 1 - SS_res / SS_tot
	return(R2,intercept,slope,resids)

def acf_vectorized(resid, nlags):
    """
    Compute autocorrelation up to nlags for multiple time series at once.
    
    resid : array of shape (T, n_series)
        Time series stacked along second axis (e.g. flattened spatial grid).
    nlags : int
        Number of lags to compute.
    
    Returns
    -------
    acf : array of shape (nlags+1, n_series)
        Autocorrelation for each lag and series.
    """
    T, n_series = resid.shape

    # Demean
    resid = resid - resid.mean(axis=0, keepdims=True)

    # FFT-based autocorrelation for efficiency
    nfft = 1 << (2*T-1).bit_length()  # next power of 2 for speed
    f = np.fft.rfft(resid, n=nfft, axis=0)
    power = f * f.conjugate()
    acov = np.fft.irfft(power, n=nfft, axis=0)[:T]

    # Normalize by variance
    acov /= acov[0]  # variance per series at lag 0

    return acov[:nlags+1]

#system inputs
model=sys.argv[1] #'GFDL-ESM4'
varn=sys.argv[2]
metric=sys.argv[3]
endyr=int(sys.argv[4])
rolling=True

#load data
climex=xr.open_dataset("clim_extremes/" + varn + "/" + model + "_" + varn + "_" + metric + ".nc")
if climex.lon.max()>180:
	climex['lon'] = (climex['lon'] + 180) % 360 - 180
	climex=climex.sortby("lon")
climex=climex.to_array(name=varn+"_exp").squeeze()
#Match exactly ERA5 time-frame 
climex=climex[:,climex.year>1939,:,:]
climex=climex.loc[:,climex.year<endyr,...]

#If heavy precip extreme intensity express as % change of baseline, for consistency with clausius clapeyron
if (varn=="pr") & (metric=="X5"):
        climex=100*climex/(climex.loc[:,climex.year<1980,:,:].mean(dim="year"))

GMT=xr.open_dataset("GMT_data/historical/" + model + "/" + model + "_GMT_historical_ssp585_allmembers.nc")
GMT=GMT["GMT"]
GMT=GMT[:,GMT.year>1939]
GMT=GMT[:,GMT.year<endyr]
if rolling:
        GMT=GMT=GMT.rolling(year=11,center=True).mean()
GMT=GMT.dropna("year")

members=GMT.member.values
years=GMT.year

climex=climex[:,climex.year.isin(years),:]

#run attribution for historical simulations of different ensemble members
climex=climex.stack(z=("lat","lon"))
climex_attr=[]
coefs_l=[]
acfs_l=[]
NL=10
for m, member in enumerate(members):
	[R2,intercept,slope,resids]=lin_reg_2D(GMT[m,:],climex[m,:,:].values)	
	coefs_l.append(slope)
	#Run autocorrelation function on this
	acf=acf_vectorized(np.array(resids),NL)
	acfs_l.append(acf)
	print("done " + member)

#Run autocorrelation function on this
acfs=np.array(acfs_l)

#output the data 
folder="/gpfs/scratch/bsc32/bsc400019/"
coefs=np.array(coefs_l)
coefs_df=xr.DataArray(data=coefs,dims=climex[:,0,:].dims,coords={**climex[:,0,:].coords},name="coef")
coefs_df=coefs_df.unstack()
coefs_df.to_netcdf(folder + "attribution/attr_coefs_" + model + "_" + varn + "_" + metric + "_" + str(endyr) + ".nc")
acfs=np.array(acfs).swapaxes(-1,-2)
acfs_df=xr.DataArray(data=acfs,dims=climex[:,0,:].dims+("lag",),coords={**climex[:,0,:].coords,"lag":[x for x in range(NL+1)]},name="acf")
acfs_df=acfs_df.unstack()
acfs_df.to_netcdf(folder + "attribution/autocorr/" + model + "_" + varn + "_" + metric + "_" + str(endyr) + ".nc")

