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

#system inputs
model=sys.argv[1] #'GFDL-ESM4'
varn=sys.argv[2]
thresh=sys.argv[3]
m=int(sys.argv[4])
agg=sys.argv[5]
membspec=int(sys.argv[6])
rolling=True

#load data
if membspec:
	climex=xr.open_dataset("clim_extremes/" + varn + "/" + model + "_" + varn + "_" + str(thresh) + "_membspec.nc")
else:
	climex=xr.open_dataset("clim_extremes/" + varn + "/" + model + "_" + varn + "_" + str(thresh) + ".nc")
if climex.lon.max()>180:
        climex['lon'] = (climex['lon'] + 180) % 360 - 180
        climex=climex.sortby("lon")
climex=climex.to_array(name=varn+"_exp").squeeze()
#Match exactly ERA5 time-frame 
climex=climex[:,climex.year>1939,:,:]
climex=climex.loc[:,climex.year<2025,...]

GMT=xr.open_dataset("GMT_data/historical/" + model + "/" + model + "_GMT_historical_ssp585_allmembers.nc")
GMT=GMT["GMT"]
GMT=GMT[:,GMT.year>1939]
GMT=GMT[:,GMT.year<2025]
if rolling:
        GMT=GMT=GMT.rolling(year=11,center=True).mean()
GMT=GMT.dropna("year")

members=GMT.member.values
years=GMT.year

climex=climex[:,climex.year.isin(years),:]

#Block bootstrap climate data, then run GMT regression/attribution
#Aggregate if necessary
if agg!="NOPE":
	deg=int(agg)
	res=np.mean(np.diff(climex.lat))
	climex=climex.coarsen(lon=round(deg/res),lat=round(deg/res),boundary="pad").mean()

climex=climex.stack(z=("lat","lon"))
climex=climex.transpose("member","z","year")
coefs=[]
resids=[]
#set number of lags for ACF function
NL=10

member=climex.member.values[m]
climex=climex[m,:,:]

for x in range(climex.shape[0]):
	
	y=climex[x,:].values
	if sum(y!=0)==0:
		coefs.append(0)
		resids.append(np.zeros_like(y))		
	else:
		X=pd.DataFrame({'gmts':GMT[m,:]})
		X=sm.add_constant(X)
		#run poisson regression with GMT for attribution
		[pred,pred_link,resid,coef]=poiss_reg(X,y)
		#store coefficients and (deviance) residuals				
		coefs.append(coef)
		resids.append(resid)

	print(str(x))
#Run assessment of autocorrelation structure on residuals for each ensemble member
resids=np.array(resids).T
acf=acf_vectorized(resids,NL)	
#store autocorrelation function

print("!!! DONE " + member + " " + str(m))				

#output the data 
coefs=np.array(coefs)
coefs_df=xr.DataArray(data=coefs,dims=climex[:,0].dims,coords={**climex[:,0].coords},name="coef")
coefs_df=coefs_df.unstack()
if membspec:
	coefs_df.to_netcdf("attribution/attr_coefs_" + model + "_" + varn + "_" + thresh + "_agg_" + agg + "_membspec_" + str(m) + ".nc")
else:
	coefs_df.to_netcdf("attribution/attr_coefs_" + model + "_" + varn + "_" + thresh + "_agg_" + agg + "_" + str(m) + ".nc")
acf=np.array(acf).swapaxes(-1,-2)
acfs_df=xr.DataArray(data=acf,dims=climex[:,0].dims+("lag",),coords={**climex[:,0].coords,"lag":[x for x in range(NL+1)]},name="acf")
acfs_df=acfs_df.unstack()
if membspec:
	acfs_df.to_netcdf("attribution/autocorr/" + model + "_" + varn + "_" + thresh + "_agg_" + agg + "_membspec_"  + str(m) + ".nc")
else:
	acfs_df.to_netcdf("attribution/autocorr/" + model + "_" + varn + "_" + thresh + "_agg_" + agg + "_"  + str(m) + ".nc")


