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
varn=sys.argv[2]
model=sys.argv[1] #'GFDL-ESM4'
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

#coordinates of Berlin, Sao Paolo, Delhi, Sydney
coords=[[52.5,13.4],[-23.56,-46.64],[28.7,77.1],[-33.9,151.2]]

[climex.sel(lat=lat, lon=lon, method="nearest") for lat, lon in coords]

results = []

for lat, lon in coords:
	da = climex.sel(lat=lat, lon=lon, method="nearest")
	results.append(da.expand_dims(points=[f"{lat}_{lon}"]))
	print(da.shape)

subset = xr.concat(results, dim="points")

#run attribution for historical simulations of different ensemble members
attribute='GMTpoiss'
year_sample=np.linspace(0,len(years)-1,len(years)).astype(int)
climex_attr=[]
coefs_l=[]
for m, member in enumerate(list(members)):
	[attr,R2s,coefs]=attribute_climate_change(subset[:,m,:],attribute,year_sample,GMT[m,:])
	climex_attr.append(attr)
	coefs_l.append(coefs)
	print("done " + member)

climex_attr=np.squeeze(np.array(climex_attr))
coefs=np.array(coefs_l)

##calculate RMSE between ensemble members prior to and after GMT corr
#rmse=RMSE(extind_membs[0],extind_membs[1],axis=-1)
#rmse_attr=RMSE(ext_attr_membs[0],ext_attr_membs[1],axis=-1)
##calculate a normalised rmse by the strength of the CC signal
#baseline_freq=365*(100-int(thresh))/10
#rmse_attr_n=np.divide(rmse_attr,abs(baseline_freq*(np.e**(np.mean(coefs,axis=0))-1)))
##calculate the correlation coefficients between ensemble members
#corr=pearson_corr(extind_membs[0],extind_membs[1],axis=-1)
#corr_attr=pearson_corr(ext_attr_membs[0],ext_attr_membs[1],axis=-1)

fs=6
cm=1/2.54
w1=6*cm
h1=5*cm

plt.close()
widths=[w1]*3
heights=[h1]*4
attribute='GMTpoiss'
fig=plt.figure(figsize=(sum(widths)+0.9,sum(heights)+0.9))
gs=fig.add_gridspec(ncols=len(widths),nrows=len(heights),width_ratios=widths,height_ratios=heights,wspace=0.7,hspace=0.3)

for i in range(4):
	#plot GMT
	ax=plt.subplot(gs[i,0])
	for m, member in enumerate(members):
		ax.plot(years,GMT[m,:],c='k',alpha=0.2)	
	if i==3:
		ax.set_xlabel('year',fontsize=fs)
	ax.set_ylabel('GMT (C)',fontsize=fs)
	ax.tick_params(axis='both', labelsize=fs)
	#plot historical 
	ax=plt.subplot(gs[i,1]) 
	for m, member in enumerate(members):
		ax.plot(years,subset[i,m,:],c='k',alpha=0.2)
	if i==3:
		ax.set_xlabel('year',fontsize=fs)
	ax.set_ylabel('Annual no. hot days (' + str(thresh) + '%-ile)',fontsize=fs)
	ax.tick_params(axis='both', labelsize=fs)
	ax.legend(fontsize=fs,loc='upper left')
	ax.set_title(str(coords[i]),fontsize=fs)
	#ax.set_title('Raw\n(corr='  + '{0:.2f}'.format(corr[ri[i],rj[i]]) + '; RMSE=' + '{0:.2g}'.format(rmse[ri[i],rj[i]]) + '; RMSE_n='+'{0:.2g}'.format(rmse_n[ri[i],rj[i]]) + ')',fontsize=fs)
	#plot after attribution
	ax=plt.subplot(gs[i,2])
	for m, member in enumerate(members):
		ax.plot(years,climex_attr[m,i,:],c='k',alpha=0.2)
	if i==3:
		ax.set_xlabel('year',fontsize=fs)
	ax.set_ylabel('Annual no. hot days (' + str(thresh) + '%-ile)',fontsize=fs)
	ax.tick_params(axis='both', labelsize=fs)
	ax.legend(fontsize=fs,loc='upper left') 

	#ax.set_title('GMT-attributed\n(corr=' + '{0:.2f}'.format(corr_attr[ri[i],rj[i]]) + '; RMSE=' + '{0:.2g}'.format(rmse_attr[ri[i],rj[i]]) + '; RMSE_n='+'{0:.2g}'.format(rmse_attr_n[ri[i],rj[i]])+')',fontsize=fs)

if rolling:
	plt.savefig('figs/member_examples/' + model + "_" + varn + '_' + thresh + '_' + attribute + '_rolling_centre_example.png',bbox_inches='tight',dpi=300)
else:
	plt.savefig('figs/member_examples/' + model + "_" + varn + '_' + thresh + '_' + attribute + '_' + '_example.png',bbox_inches='tight',dpi=300)
plt.close()


