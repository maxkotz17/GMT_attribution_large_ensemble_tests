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
		coefs_l[v].append([])
		for x in range(climexs[v].shape[-1]):
			if "exp" in metric:
				[pred,pred_link,resids,coefs]=poiss_reg(GMT[m,:],climexs[v][m,:,x,x].values)
			elif "X" in metric:
				[pred,coefs]=lin_reg(GMT[m,:],climexs[v][m,:,x,x].values)		
			climex_attr[v][m].append(pred)
			coefs_l[v][m].append(coefs)

climex_attr=np.squeeze(np.array(climex_attr))
#GMST regression slopes (variable, member, location)
coefs=np.array(coefs_l)

letters=list(string.ascii_lowercase)
fs=6
cm=1/2.54
w1=4.7*cm
h1=4.5*cm
city_labels=["Berlin (52.5°N, 13.4°E)","São Paulo (23.6°S, 46.6°W)"]

#highlight three members spanning the range of attributable change in the first local panel (Berlin temperature extremes),
#kept the same across all panels; all other members in grey, ensemble mean in black
change=climex_attr-climex_attr[...,:1]
order=np.argsort(change[0,:,0,-1])
nm=len(members)
hl=[order[int(0.1*nm)],order[nm//2],order[int(0.9*nm)]]
hl_cols=["#1b9e77","#d95f02","#7570b3"]

def plot_members(ax,x,Y):
	#Y: (member, time)
	for m in range(nm):
		ax.plot(x,Y[m],c="grey",alpha=0.15,lw=0.5)
	for k, m in enumerate(hl):
		ax.plot(x,Y[m],c=hl_cols[k],lw=0.9)
	ax.plot(x,Y.mean(axis=0),c="k",lw=1.3)

def style(ax):
	ax.tick_params(axis='both', labelsize=fs)
	ax.grid(lw=0.4,alpha=0.4)
	ax.set_axisbelow(True)

plt.close()
widths=[w1]*4
heights=[h1*1.3]+[h1]*2
fig=plt.figure(figsize=(sum(widths)+1.4,sum(heights)+1.1))
gs=fig.add_gridspec(ncols=len(widths),nrows=len(heights),width_ratios=widths,height_ratios=heights,wspace=0.6,hspace=0.45)

#a: smoothed GMST
ax=plt.subplot(gs[0,1:3])
plot_members(ax,years,GMT.values)
ax.set_xlabel('Year',fontsize=fs)
ax.set_ylabel('Global-mean surface temperature (°C)',fontsize=fs)
style(ax)
ax.annotate("a",xy=(-0.12,1.03),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")
handles=[plt.Line2D([],[],c=hl_cols[k],lw=0.9,label="Member "+str(members[m])) for k, m in enumerate(hl)]
handles+=[plt.Line2D([],[],c="grey",alpha=0.4,lw=0.5,label="Other members"),plt.Line2D([],[],c="k",lw=1.3,label="Ensemble mean")]
ax.legend(handles=handles,fontsize=fs,frameon=False,loc="upper left",bbox_to_anchor=(1.03,1))

for v, varn in enumerate(varns):
	if varn=="tas":
		name="T"
		unit="(°C)"
	elif varn=="pr":
		name="P"
		unit="(mm day$^{-1}$)"
	if metric=="X5":
		name+="5d"
		slope_unit="("+unit[1:-1]+" K$^{-1}$)"
	elif metric=="0.99_exp":
		name+="99p"
		unit="(days)"
		slope_unit="(K$^{-1}$)"

	for i in range(2):
		#b-e: local climate extremes
		ax=plt.subplot(gs[1,v*2+i])
		plot_members(ax,years,climexs[v][:,:,i,i].values)
		ax.set_xticks([1960,2000])
		ax.set_xlabel('Year',fontsize=fs)
		if i==0:
			ax.set_ylabel(name+" "+unit,fontsize=fs)
		style(ax)
		ax.set_title(city_labels[i],fontsize=fs)
		ax.annotate(letters[1+v*2+i],xy=(-0.12,1.05),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

		#f-i: change correlated with GMST since the first year (fitted values relative to the first year), with the
		#distribution of the GMST regression slope across members on the right and its relative uncertainty (eq. 3)
		ax=plt.subplot(gs[2,v*2+i])
		Y=change[v,:,i,:]
		plot_members(ax,years,Y)
		ax.axhline(0,c="k",lw=0.5,ls=":")
		ax.set_xticks([1960,2000])
		ax.set_xlabel('Year',fontsize=fs)
		if i==0:
			ax.set_ylabel("Attributable change in\n"+name+" since "+str(int(years[0]))+" "+unit,fontsize=fs)
		style(ax)
		ax.annotate(letters[5+v*2+i],xy=(-0.12,1.05),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")
		beta=coefs[v,:,i]
		#slopes as estimated (Poisson: change in log frequency per K), on which chi is computed
		b=beta
		chi=100*beta.std()/np.abs(beta.mean())
		ax.annotate(r"$\chi$ = "+str(int(round(chi)))+"%",xy=(0.04,0.9),xycoords="axes fraction",fontsize=fs)
		axh=make_axes_locatable(ax).append_axes("right",size="28%",pad=0.05)
		axh.hist(b,bins=15,orientation="horizontal",color="grey",alpha=0.5)
		for k, m in enumerate(hl):
			axh.axhline(b[m],c=hl_cols[k],lw=0.9)
		axh.axhline(b.mean(),c="k",lw=1.3)
		axh.axhline(0,c="k",lw=0.5,ls=":")
		axh.yaxis.tick_right()
		axh.yaxis.set_label_position("right")
		axh.set_title(r"$\beta$"+"\n"+slope_unit,fontsize=fs-1)
		axh.tick_params(axis="both",labelsize=fs-1)
		axh.set_xlabel("Members",fontsize=fs-1)
		print(name,city_labels[i],"chi %.0f%%"%chi)

if membspec:
	plt.savefig('figs/member_examples/' + model + "_taspr_" + metric + '_' + '_example_membspec_v2.png',bbox_inches='tight',dpi=300)
else:
	plt.savefig('figs/member_examples/' + model + "_taspr_" + metric + '_' + '_example_v2.png',bbox_inches='tight',dpi=300)
plt.close()
print("highlighted members:",[str(members[m]) for m in hl])
