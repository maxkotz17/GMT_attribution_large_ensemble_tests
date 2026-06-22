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

#attribute changes in local climate conditions to global climate, either via regression with global temperature, or random sampling the first thirty years
def attribute_climate_change(CL,attribute,year_sample,gmt):
	if attribute=='resample':
		CS=CL[:,year_sample]
		CA=CL-CS
	else:
		#sample years available for climate data to get estimate on uncertainty of attribution 
		CA=np.zeros_like(CL)
		R2s=np.zeros((CL.shape[0]))
		coefs=np.zeros_like(R2s)
		if attribute=='GMTlin':
			for x in range(CL.shape[0]): 
				beta=simple_linear_regression(gmt,CL[x,:])
				pred=gmt*beta[1]+beta[0]
				CA[x,:]=pred#-np.mean(pred[:30])
				coefs[x]=beta[1]
		elif attribute=='GMTpoiss':
			for x in range(CL.shape[0]): 
				y=CL[x,:]
				if sum(y!=0)==0:
					coefs[x]=0
					CA[x,:]=0
				else:
					X=pd.DataFrame({'gmts':gmt})
					X=sm.add_constant(X)
					model=sm.GLM(y,X,family=sm.families.Poisson()).fit()
					X=pd.DataFrame({'gmts':gmt})
					X=sm.add_constant(X)
					pred=model.predict(X)
					R2s[x]=model.pseudo_rsquared()
					coefs[x]=model.params[1]
					CA[x,:]=pred#-np.mean(pred[:30])
		elif attribute=='GMTnegbin':
			for x in range(CL.shape[0]): 
				y=CL[x,:]
				X=pd.DataFrame({'gmts':gmt})
				X=sm.add_constant(X)
				model=sm.GLM(y,X,family=sm.families.NegativeBinomial()).fit()
				X=pd.DataFrame({'gmts':gmt})
				X=sm.add_constant(X)
				pred=model.predict(X)
				R2s[x]=model.pseudo_rsquared()
				coefs[x]=model.params[1]
				CA[x,:]=pred#-np.mean(pred[:30])
		else:
			return('Incorrect attribution method specified')
	return(CA,R2s,coefs)

#attribute changes in local climate conditions to global climate, either via regression with global temperature, or random sampling the first thirty years
def attribute_climate_changeOLD(CL,attribute,year_sample,gmt):
    if attribute=='resample':
        CS=CL[:,year_sample]
        CA=CL-CS
    else:
        #sample years available for climate data to get estimate on uncertainty of attribution 
        CS=CL[:,year_sample]
        gmts=gmt[year_sample]
        CA=np.zeros_like(CL)
        R2s=np.zeros((CL.shape[0]))
        coefs=np.zeros_like(R2s)
        if attribute=='GMTlin':
           for x in range(CL.shape[0]):
                beta=simple_linear_regression(gmts,CS[x,:])
                pred=gmt*beta[1]+beta[0]
                CA[x,:]=pred#-np.mean(pred[:30])
                coefs[x]=beta[1]
        elif attribute=='GMTpoiss':
            for x in range(CL.shape[0]):
                y=CS[x,:]
                X=pd.DataFrame({'gmts':gmts})
                X=sm.add_constant(X)
                model=sm.GLM(y,X,family=sm.families.Poisson()).fit()
                X=pd.DataFrame({'gmts':gmt})
                X=sm.add_constant(X)
                pred=model.predict(X)
                R2s[x]=model.pseudo_rsquared()
                coefs[x]=model.params[1]
                CA[x,:]=pred#-np.mean(pred[:30])
        elif attribute=='GMTnegbin':
            for x in range(CL.shape[0]):
                y=CS[x,:]
                X=pd.DataFrame({'gmts':gmts})
                X=sm.add_constant(X)
                model=sm.GLM(y,X,family=sm.families.NegativeBinomial()).fit()
                X=pd.DataFrame({'gmts':gmt})
                X=sm.add_constant(X)
                pred=model.predict(X)
                R2s[x]=model.pseudo_rsquared()
                coefs[x]=model.params[1]
                CA[x,:]=pred#-np.mean(pred[:30])
        else:
            return('Incorrect attribution method specified')
    return(CA,R2s,coefs)

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
thresh=sys.argv[3]
rolling=True

#load data
climex=xr.open_dataset("clim_extremes/" + varn + "/" + model + "_" + varn + "_" + str(thresh) + "_exp.nc")
if climex.lon.max()>180:
        climex['lon'] = (climex['lon'] + 180) % 360 - 180
        climex=climex.sortby("lon")
climex=climex.to_array(name=varn+"_exp").squeeze()
climex=climex[:,climex.year>1939,:,:]

GMT=xr.open_dataset("GMT_data/historical/" + model + "/" + model + "_GMT_historical_ssp585_allmembers.nc")
GMT=GMT["GMT"]
if rolling:
        GMT=GMT=GMT.rolling(year=11,center=True).mean()
GMT=GMT.loc[:,GMT.year>1939]
GMT=GMT.loc[:,GMT.year<2021]
climex=climex.loc[:,climex.year<2021,...]
members=GMT.member.values
years=GMT.year

#poisson regression from statsmodels
def poiss_reg(X,y):
	X=sm.add_constant(X)
	model=sm.GLM(y,X,family=sm.families.Poisson()).fit()
	pred=model.fittedvalues
	#predictions and residuals on the link scale
	pred_link=model.fittedvalues.apply(lambda mu: np.log(mu))
	resid=model.resid_deviance.copy()
	return(pred,pred_link,resid,model.params[1])

#run first off GMT correlation, store predictions, residuals and coefficient 
climex=climex.stack(z=("lat","lon"))
preds=[]
pred_links=[]
coefs=[]
resids=[]
for m, member in enumerate(members[:1]):
	CL=climex[m,...]
	coefs.append([])
	preds.append([])
	pred_links.append([])
	resids.append([])
	for x in range(CL[:,:100].shape[1]):
		
		y=CL[:,x]
		if sum(y!=0)==0:
			coefs[m].append(0)
			resids[m].append(y)
			pred[m].append(y)	
		else:
			X=pd.DataFrame({'gmts':GMT[m,:]})
			X=sm.add_constant(X)
			[pred,pred_link,resid,coef]=poiss_reg(X,y)
				
			coefs[m].append(coef)
			preds[m].append(pred)
			pred_links[m].append(pred_link)
			resids[m].append(resid)

	print("done " + member)

#Boostrap residuals
resids=np.array(resids)
preds=np.array(preds)
pred_links=np.array(pred_links)
coefs=np.array(coefs)

#block bootstrap resample with block of width W, preserving structure (spatial) along the first dimensions
def block_resample(X,W):
	n = X.shape[-1]
	n_blocks = int(np.ceil(n / W))
	starts = np.random.randint(0, n - W + 1, size=n_blocks)
	blocks = [X[s:s + W] for s in starts]
	resampled = np.concatenate(blocks,axis=-1)
	return resampled[:n]

resids_resamp=block_resample(resids,2)

#Generate synthetic time-series
link_perturbed=pred_links+resids_resamp
perturbed=np.random.poisson(np.exp(link_perturbed))

climex_attr=climex.copy(data=np.array(climex_attr))
climex_attr=climex_attr.unstack()
coefs=np.array(coefs)
coefs_df=climex[:,0,:].copy(data=coefs)
coefs_df=coefs_df.unstack()

climex_attr.to_netcdf("attribution/attr_ts_" + model + "_" + varn + "_" + thresh + "_rollingcentre.nc")
coefs_df.to_netcdf("attribution/attr_coefs_" + model + "_" + varn + "_" + thresh + "_rollingcentre.nc")




