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

#system inputs
model=sys.argv[1] #'GFDL-ESM4'
varn=sys.argv[2]
metric=sys.argv[3]
seed=int(sys.argv[4])
N=int(sys.argv[5])
W=int(sys.argv[6])
agg=sys.argv[7]
membspec=bool(int(sys.argv[8]))

outfolder="/gpfs/scratch/bsc32/bsc400019/"

rolling=True

#load data
#load data
if membspec:
        climex=xr.open_dataset("clim_extremes/" + varn + "/" + model + "_" + varn + "_" + metric + "_membspec.nc")
else:
        climex=xr.open_dataset("clim_extremes/" + varn + "/" + model + "_" + varn + "_" + metric + ".nc")

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

#Aggregate if necessary
if agg!="NOPE":
        deg=int(agg)
        res=np.mean(np.diff(climex.lat))
        climex=climex.coarsen(lon=round(deg/res),lat=round(deg/res),boundary="pad").mean()

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

#Block bootstrap climate data, then run GMT regression/attribution
#first subsample the data for run-time processes for now
climex=climex.stack(z=("lat","lon"))
climex=climex.transpose("member","z","year")

GMT_expanded = GMT.expand_dims({"z": 1})

# Create a MultiIndex with lat and lon
multi_idx = pd.MultiIndex.from_arrays(
    [[np.nan], [np.nan]], 
    names=['lat', 'lon']
)

# Assign the MultiIndex to the z coordinate
GMT_expanded = GMT_expanded.assign_coords(z=multi_idx)

preds=[]
coefs=[]
np.random.seed(seed)

for n in range(N):
	#run common block resample on both the whole climate data and GMT, consistently across ensemble members
	CL=block_resample(xr.concat((climex,GMT_expanded),dim="z"),W)
	GMT_r=CL[:,-1,:]
	CL=CL[:,:-1,:]

	coefs.append([])
	preds.append([])
	for m, member in enumerate(members):
		coefs[n].append([])
		preds[n].append([])
	
		for x in range(CL.shape[1]):
			
			y=CL[m,x,:]
			if sum(y!=0)==0:
				coefs[n][m].append(0)	
				preds[n][m].append(y)	
			else:
				X=pd.DataFrame({'gmts':GMT_r[m,:]})
				X=sm.add_constant(X)
				#run poisson regression with GMT for attribution
				[pred,pred_link,resid,coef]=poiss_reg(X,y)
					
				coefs[n][m].append(coef)
				#preds[n][m].append(pred)

		if m%10==0:
			print("!!! DONE " + member + " " + str(m))				
	print("done " + str(n))

#output the data 
coefs=np.array(coefs)
coefs_df=xr.DataArray(data=coefs,dims=("sample",)+climex[:m+1,:,0].dims,coords={"sample":[x for x in range(N)],**climex[:m+1,:,0].coords},name="coef")
coefs_df=coefs_df.unstack()
metric+="_agg_" + agg
if membspec:
	metric+="_membspec"
coefs_df.to_netcdf(outfolder + "attribution/bootstrap/attr_coefs_" + model + "_" + varn + "_" + metric + "_block" + str(W) + "_seed" + str(seed) + ".nc")


