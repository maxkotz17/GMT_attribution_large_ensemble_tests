import xarray as xr
import numpy as np
import pandas as pd
import scipy.interpolate
import sys
import statsmodels.api as sm
import statsmodels.formula.api as smf
from multiprocessing import Pool, cpu_count

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
ntasks=int(sys.argv[4])
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

#scaling down for debugging
#GMT=GMT[:12,:]
#climex=climex[:12,:,:5,:5]

###Parallel processing of members
def split_data(GMT, climex, nchunks):
	split_indices=np.array_split(GMT.member.values, nchunks)
	return([GMT.sel({"member":idxs}) for idxs in split_indices],[climex.sel({"member":idxs}) for idxs in split_indices])

def process_wrapper(GMTchunk,climexchunk):
	return process_chunk(GMTchunk,climexchunk)

def process_chunk(GMTchunk,climexchunk):
	members=GMTchunk.member.values
	
	climexchunk=climexchunk.stack(z=("lat","lon"))
	attribute='GMTpoiss'
	years=GMT.year
	year_sample=np.linspace(0,len(years)-1,len(years)).astype(int)

	climex_attr=[]
	coefs_l=[]
	for m, member in enumerate(members):
		[attr,R2s,coefs]=attribute_climate_change(climexchunk[m,:,...].T,attribute,year_sample,GMTchunk[m,:])
		climex_attr.append(attr.T)
		coefs_l.append(coefs)
		print("done " + member)

	climex_attr=climexchunk.copy(data=np.array(climex_attr))
	climex_attr=climex_attr.unstack()
	coefs=np.array(coefs_l)
	coefs_df=climexchunk[:,0,:].copy(data=coefs)
	coefs_df=coefs_df.unstack()

	return(climex_attr,coefs_df)

if __name__ == '__main__':
	nchunks=min(ntasks,cpu_count())
	[GMTchunks,climexchunks]=split_data(GMT,climex,nchunks)

	with Pool(processes=nchunks) as pool:
		output = pool.starmap(process_wrapper,[(GMTchunks[i], climexchunks[i]) for i in range(len(GMTchunks))])

climex_attr = xr.concat([output[n][0] for n in range(ntasks)],dim="member")
coefs_df = xr.concat([output[n][1] for n in range(ntasks)],dim="member")
	

climex_attr.to_netcdf("attribution/attr_ts_" + model + "_" + varn + "_" + thresh + "_rollingcentre.nc")
coefs_df.to_netcdf("attribution/attr_coefs_" + model + "_" + varn + "_" + thresh + "_rollingcentre.nc")


