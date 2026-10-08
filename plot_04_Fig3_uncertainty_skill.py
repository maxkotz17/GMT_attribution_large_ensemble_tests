import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import sys
import glob

#Figure 3 (revised): skill of the block-bootstrapped relative uncertainty against the large ensembles
#top row: area below a given relative uncertainty (as previous Fig. 3a), precision of identifying that area
#bottom row: pattern correlation with the own large ensemble (as previous Fig. 3b), and with each model's large ensemble
#inputs from 17_bootstrap_skill_stats.py

varn=sys.argv[1] #tas
metric=sys.argv[2] #0.99_exp_agg_NOPE_membspec or X5
W=int(sys.argv[3]) #5
relerr_cutoff=int(sys.argv[4]) #100 for tas, 200 for pr
ERA5=bool(int(sys.argv[5]))
#optional end year suffix of the coefficient files (e.g. 2025 for pr X5)
endyr="_"+sys.argv[6] if len(sys.argv)>6 else ""

models=["MIROC6","MPI-ESM1-2-LR","CanESM5"]
cols=["tab:blue","tab:orange","tab:green"]
folder="/gpfs/scratch/bsc32/bsc400019/"
skill=folder+"attribution/skill/"
tag=varn+"_"+metric+"_block"+str(W)+"_"+str(relerr_cutoff)+endyr

if varn=="tas":
	relerrs=[(x+1)*10 for x in range(10)]
else:
	relerrs=[(x+1)*10 for x in range(20)]
#precision only shown where at least 1% of the area is truly below the threshold in the large ensemble
min_base=0.01

#area fraction with relative uncertainty below each threshold, area weighted (non-finite values count as above), as in plot_04
def area_frac(chi):
	w=np.cos(np.deg2rad(chi.lat)).broadcast_like(chi)
	return(np.array([float(w.where(chi<r,0).sum())/float(w.sum()) for r in relerrs]))

def area_frac_members(chi):
	w=np.cos(np.deg2rad(chi.lat))
	return(np.array([[float((chi[n]<r).weighted(w).mean()) for r in relerrs] for n in range(chi.member.size)]))

af_LE=[]
af_BS=[]
for model in models:
	bias=xr.open_dataset(skill+"bias_maps_"+model+"_"+tag+".nc")
	af_LE.append(area_frac(bias.chi_LE))
	af_BS.append(area_frac_members(xr.open_dataset(skill+"chi_bootstrap_"+model+"_"+tag+".nc").chi))
af_LE=np.array(af_LE)
af_BS=np.array(af_BS)
#legends where the curves are not: lower right when the area curves rise early (most area below X), else upper left
leg_loc="lower right" if np.median(af_LE[:,len(relerrs)//2])>0.5 else "upper left"

if ERA5:
	#ERA5 bootstrap, file conventions as in plot_04
	if "X" in metric:
		E5=xr.open_dataset(folder + "attribution/bootstrap/attr_coefs_ERA5_" + varn + "_" + metric.split("X5")[0] + "X5" + "_block" + str(W) + ".nc")
	else:
		files=glob.glob(folder + "attribution/bootstrap/attr_coefs_ERA5_" + varn + "_0.99_block"+str(W)+"*.nc")
		E5=xr.concat([xr.open_dataset(f) for f in files],dim="sample")
	E5_chi=100*E5.coef.std(dim="sample")/np.abs(E5.coef.mean(dim="sample"))
	af_E5=area_frac(E5_chi)

thr=pd.read_csv(skill+"threshold_skill_"+tag+".csv")
obs=pd.read_csv(skill+"obs_skill_"+tag+".csv")
#pattern correlations with all capped maps (large ensembles and bootstrapped members) regridded to the common CanESM5 grid
cross=pd.read_csv(skill+"cross_model_common_grid_"+tag+".csv")

fs=6
cm=1/2.54
w1=6*cm
h1=5.5*cm

def style(ax):
	ax.tick_params(axis='both', which='major', labelsize=fs)
	ax.grid(which="major",axis="both",lw=0.5,alpha=0.5)
	ax.set_axisbelow(True)

def annotate_panel(ax,letter,x=-0.18):
	ax.annotate(letter,xy=(x,1.06),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

plt.close()
fig=plt.figure(figsize=(2*w1+1.0,2*h1+1.0))
gs0=fig.add_gridspec(nrows=2,hspace=0.6)
gs_top=gs0[0].subgridspec(1,2,wspace=0.5)
#bottom row needs more space for the matrix row labels
gs_bot=gs0[1].subgridspec(1,2,wspace=0.85)
xlim=[relerrs[0],relerrs[-1]]
#panels a-b share the same x-axis: thresholds of relative uncertainty due to internal variability
xlab="Relative uncertainty due to\ninternal variability, X (%)"
model_handles=[Line2D([],[],c=cols[m],lw=1.5,label=models[m]) for m in range(3)]

#a: cumulative area below a given relative uncertainty, large ensemble vs. bootstrapped members
ax=fig.add_subplot(gs_top[0])
for m, model in enumerate(models):
	ax.plot(relerrs,af_LE[m],c=cols[m],lw=1.5)
	ax.plot(relerrs,np.median(af_BS[m],axis=0),c=cols[m],ls="--",lw=1.5)
	ax.fill_between(relerrs,np.percentile(af_BS[m],5,axis=0),np.percentile(af_BS[m],95,axis=0),color=cols[m],alpha=0.15,lw=0)
#line styles here; model colours in panel b for tas, here for pr
handles=[Line2D([],[],c="grey",lw=1.5,label="Large ensemble"),Line2D([],[],c="grey",lw=1.5,ls="--",label="Bootstrap (median, 5-95%)" if varn=="tas" else "Bootstrap\n(median, 5-95%)")]
if varn!="tas":
	handles=model_handles+handles
if ERA5:
	ax.plot(relerrs,af_E5,c="k",ls="--",lw=1.5)
	handles.append(Line2D([],[],c="k",lw=1.5,ls="--",label="ERA5: bootstrap"))
ax.legend(handles=handles,fontsize=fs-1,loc=leg_loc,frameon=False)
ax.set_xlabel(xlab,fontsize=fs)
ax.set_ylabel("Fraction of global surface area below X",fontsize=fs)
ax.set_xlim(xlim)
ax.set_ylim([0,1])
style(ax)
annotate_panel(ax,"a")

#b: precision of identifying the area below a given relative uncertainty
ax=fig.add_subplot(gs_top[1])
for m, model in enumerate(models):
	dm=thr[thr.model==model]
	keep=dm.groupby("thresh").base_rate.median()>=min_base
	d=dm[dm.thresh.isin(keep.index[keep])].groupby("thresh")
	x=np.array(list(d.groups.keys()))
	ax.plot(x,d["precision"].median(),c=cols[m],lw=1.5)
	ax.fill_between(x,d["precision"].quantile(0.05),d["precision"].quantile(0.95),color=cols[m],alpha=0.15,lw=0)
	#no skill: precision of a random guess = area fraction truly below X (solid lines in a)
	ax.plot(x,d["base_rate"].median(),c=cols[m],lw=1,ls=":")
noskill=[Line2D([],[],c="grey",lw=1,ls=":",label="Random guess")]
ax.legend(handles=(model_handles if varn=="tas" else [])+noskill,fontsize=fs-1,loc=leg_loc,frameon=False)
ax.set_xlabel(xlab,fontsize=fs)
ax.set_ylabel("Precision: fraction of area identified\nbelow X that is truly below X",fontsize=fs)
ax.set_xlim(xlim)
ax.set_ylim([0,1])
style(ax)
annotate_panel(ax,"b")

#c: pattern correlation between bootstrapped members and their own large ensemble, as previous Fig. 3b
ax=fig.add_subplot(gs_bot[0])
#own-model pattern correlations on each model's native grid (as quoted in the text); d uses the common CanESM5 grid
df=obs.rename(columns={"r_obs":"Pattern cor"})
sns.violinplot(data=df,x="model",y="Pattern cor",order=models,ax=ax,palette=cols,hue="model",hue_order=models,legend=False,linewidth=0.6)
ax.set_ylim([0,1])
ax.set_ylabel("Weighted pattern correlation between\nlarge ensemble and bootstrapped members",fontsize=fs)
ax.set_xlabel("Model",fontsize=fs)
plt.setp(ax.get_xticklabels(),rotation=20,ha="right")
style(ax)
annotate_panel(ax,"c",x=-0.32)

#d: median pattern correlation with each model's large ensemble (rows: bootstrapped model, evaluated on its own grid)
ax=fig.add_subplot(gs_bot[1])
mat=cross.groupby(["model_bs","model_le"])["r"].median().unstack().loc[models,models]
sns.heatmap(mat,ax=ax,cmap="Blues",annot=True,fmt=".2f",annot_kws={"fontsize":fs},linewidths=1,linecolor="white",square=True,
	cbar_kws={"shrink":0.8},xticklabels=models,yticklabels=models)
ax.collections[0].colorbar.ax.tick_params(labelsize=fs)
ax.collections[0].colorbar.set_label("Median pattern correlation of\n"+r"relative uncertainty ($\chi$)",fontsize=fs)
for k in range(3):
	ax.add_patch(plt.Rectangle((k,k),1,1,fill=False,ec="k",lw=1))
ax.set_xlabel("Large ensemble (reference)",fontsize=fs)
ax.set_ylabel("Model of bootstrapped member",fontsize=fs)
ax.tick_params(axis='both', which='major', labelsize=fs,length=0)
plt.setp(ax.get_xticklabels(),rotation=20,ha="right")
plt.setp(ax.get_yticklabels(),rotation=0)
own=cross[cross.model_bs==cross.model_le]
print("members closest to own model by pattern correlation: "+str(int(own["best_r"].sum()))+"/"+str(len(own)))
ax.annotate("d",xy=(-0.55,1.06),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

name="figs/member_stats/" + varn + "_" + metric + "_uncertainty_skill_bootens_block" + str(W) + "_" + str(relerr_cutoff) + endyr + ("_ERA5" if ERA5 else "") + ".png"
plt.savefig(name,bbox_inches="tight",dpi=300)
plt.close()
print("saved " + name)
print("panel c (native grid) medians:",obs.groupby("model").r_obs.median().round(3).to_dict())
print("panel d (common grid) diagonal:",cross[cross.model_bs==cross.model_le].groupby("model_bs").r.median().round(3).to_dict())
