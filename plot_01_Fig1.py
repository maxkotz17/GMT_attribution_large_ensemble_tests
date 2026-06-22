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
import string

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
        pred_link=np.log(model.fittedvalues)
        resid=model.resid_deviance.copy()
        return(pred,pred_link,resid,model.params[1])

def lin_reg(X,y):
	X=sm.add_constant(X)
	model=sm.OLS(y,X).fit()
	pred=model.fittedvalues
	return(pred,model.params[1])	

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

#system inputs
model=sys.argv[1] #'GFDL-ESM4'
metric=sys.argv[2]
membspec=bool(int(sys.argv[3]))
rolling=True

GMT=xr.open_dataset("GMT_data/historical/" + model + "/" + model + "_GMT_historical_ssp585_allmembers.nc")
GMT=GMT["GMT"]
GMT=GMT.loc[:,GMT.year>1939]
GMT=GMT.loc[:,GMT.year<2025]
if rolling:
        GMT=GMT=GMT.rolling(year=11,center=True).mean()

GMT=GMT.dropna("year")
members=GMT.member.values
years=GMT.year

varns=["tas","pr"]
#coordinates of Berlin, Sao Paolo, Delhi, Sydney
cities=["Berlin","Sao Paolo","Delhi","Sydney"]
coords=[[52.5,13.4],[-23.56,-46.64],[28.7,77.1],[-33.9,151.2]]
lats=[52.5,-23.56,28.7,-33.9]
lons=[13.4,-46.64,77.1,151.2]
climexs=[]
for v, varn in enumerate(varns):
	#load data
	if membspec:
		climex=xr.open_dataset("clim_extremes/" + varn + "/" + model + "_" + varn + "_" + metric + "_membspec.nc")
	else:
		climex=xr.open_dataset("clim_extremes/" + varn + "/" + model + "_" + varn + "_" + metric + ".nc")
	if climex.lon.max()>180:
		climex['lon'] = (climex['lon'] + 180) % 360 - 180
		climex=climex.sortby("lon")
	climex=climex.to_array(name=varn+"_exp").squeeze()
	climex=climex[:,climex.year.isin(years),:,:]
	climexs.append(climex.sel(lat=lats,lon=lons,method="nearest"))	

#run attribution for historical simulations of different ensemble members
climex_attr=[]
coefs_l=[]
for v, varn in enumerate(varns):	
	coefs_l.append([])
	climex_attr.append([])
	for m, member in enumerate(list(members)):
		climex_attr[v].append([])
		for x in range(climexs[v].shape[-1]):
			if "exp" in metric:
				[pred,pred_link,resids,coefs]=poiss_reg(GMT[m,:],climexs[v][m,:,x,x].values)
			elif "X" in metric:
				[pred,coefs]=lin_reg(GMT[m,:],climexs[v][m,:,x,x].values)		
			climex_attr[v][m].append(pred)
			#coefs_l.append(coefs[1])
			print("done " + member)

climex_attr=np.squeeze(np.array(climex_attr))
coefs=np.array(coefs_l)

letters=list(string.ascii_lowercase)
fs=6
cm=1/2.54
w1=4.7*cm
h1=4.5*cm

plt.close()
widths=[w1]*4
heights=[h1*1.5]+[h1]*2
cols=["tab:red","tab:blue"]
fig=plt.figure(figsize=(sum(widths)+1.2,sum(heights)+1.1))
gs=fig.add_gridspec(ncols=len(widths),nrows=len(heights),width_ratios=widths,height_ratios=heights,wspace=0.4,hspace=0.3)

#plot GMT
ax=plt.subplot(gs[0,1:3])
for m, member in enumerate(members):
	if m<5:
	        ax.plot(years,GMT[m,:],alpha=0.2,label=member)
	else:
		ax.plot(years,GMT[m,:],alpha=0.2)
ax.set_xlabel('Year',fontsize=fs)
ax.set_ylabel('Global-mean surface temperature (C)',fontsize=fs)
ax.tick_params(axis='both', labelsize=fs)
#ax.legend(fontsize=fs)
ax.annotate("a",xy=(-0.05,1.05),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

for v, varn in enumerate(varns):
	if varn=="tas":
		name="T"
		unit="(C)"
	elif varn=="pr":
		name="P"
		unit="(mm)"
	if metric=="X5":
		name+="X5d " + unit
	elif metric=="0.99_exp":
		name+="99p (days)"	

	for i in range(2):

		#plot historical 
		ax=plt.subplot(gs[1,v*2+i]) 
		for m, member in enumerate(members):
			ax.plot(years,climexs[v][m,:,i,i],alpha=0.2)

		ax.set_xlabel('Year',fontsize=fs)
		if i==0:
			ax.set_ylabel(name,fontsize=fs)
		ax.tick_params(axis='both', labelsize=fs)
		ax.set_title(cities[i] + " " + str(coords[i]),fontsize=fs)
		ax.annotate(letters[1+v*2+i],xy=(-0.1,1.05),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

		#plot after attribution
		ax=plt.subplot(gs[2,v*2+i])
		for m, member in enumerate(members):
			ax.plot(years,climex_attr[v,m,i,:],alpha=0.2)

		ax.set_xlabel('Year',fontsize=fs)
		if i==0:
		       ax.set_ylabel(name + " - attributable",fontsize=fs)
		ax.tick_params(axis='both', labelsize=fs)
		ax.annotate(letters[5+v*2+i],xy=(-0.1,1.05),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

if membspec:
	plt.savefig('figs/member_examples/' + model + "_taspr_" + metric + '_' + '_example_membspec.png',bbox_inches='tight',dpi=300)
else:
	plt.savefig('figs/member_examples/' + model + "_taspr_" + metric + '_' + '_example.png',bbox_inches='tight',dpi=300)
plt.close()


