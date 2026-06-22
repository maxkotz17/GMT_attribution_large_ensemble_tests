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

def simple_linear_regression(x, y):
    # Add a constant term to the input for the intercept
    X = np.column_stack((np.ones_like(x), x))

    # Calculate the coefficients using the normal equation
    beta = np.linalg.inv(X.T @ X) @ X.T @ y

    return beta

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

#poisson regression from statsmodels
def poiss_reg(X,y):
        X=sm.add_constant(X)
        model=sm.GLM(y,X,family=sm.families.Poisson()).fit()
        pred=model.fittedvalues
        #predictions and residuals on the link scale
        pred_link=model.fittedvalues.apply(lambda mu: np.log(mu))
        resid=model.resid_deviance.copy()
        return(pred,pred_link,resid,model.params[1])

#block bootstrap resample with block of width W, preserving structure (spatial) along the first dimensions
def block_resample(X,W):
        n = X.shape[-1]
        n_blocks = int(np.ceil(n / W))
        starts = np.random.randint(0, n - W + 1, size=n_blocks)
        blocks = [X[...,s:s + W] for s in starts]
        resampled = np.concatenate(blocks,axis=-1)
        return resampled[...,:n]

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
varn=sys.argv[1]
metric=sys.argv[2]
rolling=True

#load data
climex=xr.open_dataset("clim_extremes/" + varn + "/ERA5_" + varn + "_" + metric + ".nc")
if climex.lon.max()>180:
        climex['lon'] = (climex['lon'] + 180) % 360 - 180
        climex=climex.sortby("lon")
climex=climex.to_array(name=varn+"_exp").squeeze()

GMT=xr.open_dataset("GMT_data/ERA5_tas.nc")
GMT=GMT["GMT"]
if rolling:
        GMT=GMT=GMT.rolling(year=11,center=True).mean()
GMT=GMT.loc[GMT.year>1939]
GMT=GMT.loc[GMT.year<2021]
climex=climex.loc[climex.year<2021,...]
years=GMT.year

#Block bootstrap climate data, then run GMT regression/attribution
#first subsample the data for run-time processes for now
climex=climex[:,:,:]
climex=climex.stack(z=("lat","lon"))
climex=climex.transpose("z","year")

climex=climex[:,np.logical_not(np.isnan(GMT))]
GMT=GMT[np.logical_not(np.isnan(GMT))]

#set number of lags for ACF function
NL=10

#run poisson regression with GMT for attribution
[R2,intercepts,coefs,resids]=lin_reg_2D(GMT,climex.T)

#Run assessment of autocorrelation structure on residuals for each ensemble member
resids=np.array(resids).T
acf=acf_vectorized(resids.T,NL)	
#store autocorrelation function

#output the data 
folder="/gpfs/scratch/bsc32/bsc400019/"
coefs=np.array(coefs)
coefs_df=xr.DataArray(data=coefs,dims=climex[:,0].dims,coords={**climex[:,0].coords},name="coef")
coefs_df=coefs_df.unstack()
coefs_df.to_netcdf(folder + "attribution/attr_coefs_ERA5_" + varn + "_" + metric + ".nc")
acfs_df=xr.DataArray(data=acf.T,dims=climex[:,0].dims+("lag",),coords={**climex[:,0].coords,"lag":[x for x in range(NL+1)]},name="acf")
acfs_df=acfs_df.unstack()
acfs_df.to_netcdf(folder + "attribution/autocorr/ERA5_" + varn + "_" + metric + ".nc")


