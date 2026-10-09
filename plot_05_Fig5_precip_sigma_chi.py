import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import sys

#Figure 5 (revised): the block-bootstrap captures absolute uncertainty for precipitation extremes, while relative
#uncertainty from a single realisation saturates at low signal-to-noise
#a: area below a given relative uncertainty for precipitation extreme frequency (as previous Fig. 5a)
#b: pattern correlations of bootstrapped absolute (sigma) and relative (chi) uncertainty with the large ensemble, all metrics
#c: median pattern correlation of bootstrapped sigma maps with each model's large ensemble (common CanESM5 grid)
#d: bootstrapped against large-ensemble relative uncertainty, binned by the large-ensemble value (as Fig. A17b)
#inputs from 17_bootstrap_skill_stats.py, 18_sigma_chi_coverage.py and 19_sigma_cross_model.py

#metric of panels a and c
varn=sys.argv[1] if len(sys.argv)>1 else "pr"
metric=sys.argv[2] if len(sys.argv)>2 else "0.99_exp_agg_NOPE_membspec"
W=int(sys.argv[3]) if len(sys.argv)>3 else 5
relerr_cutoff=int(sys.argv[4]) if len(sys.argv)>4 else 200
endyr="_"+sys.argv[5] if len(sys.argv)>5 else ""

models=["MIROC6","MPI-ESM1-2-LR","CanESM5"]
cols=["tab:blue","tab:orange","tab:green"]
skill="/gpfs/scratch/bsc32/bsc400019/attribution/skill/"
tag=varn+"_"+metric+"_block"+str(W)+"_"+str(relerr_cutoff)+endyr
#metrics of panels b and d: label, file tag, colour (warm: temperature, cool: precipitation)
metrics=[(r"$T5d$","tas_X5_block5_100","#b2182b"),
	(r"$T99p$","tas_0.99_exp_agg_NOPE_membspec_block5_100","#ef8a62"),
	(r"$P5d$","pr_X5_block5_200_2025","#2166ac"),
	(r"$P99p$","pr_0.99_exp_agg_NOPE_membspec_block5_200","#67a9cf"),
	(r"$P5d$"+"\n(1940-2100)","pr_X5_block5_200_2100","#542788")]

relerrs=[(x+1)*10 for x in range(20)]

fs=6
cm=1/2.54

def style(ax):
	ax.tick_params(axis='both', which='major', labelsize=fs)
	ax.grid(which="major",axis="both",lw=0.5,alpha=0.5)
	ax.set_axisbelow(True)

def annotate_panel(ax,letter,x=-0.18):
	ax.annotate(letter,xy=(x,1.06),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

def wquantile(x,w,q):
	o=np.argsort(x)
	c=np.cumsum(w[o])/w.sum()
	return(np.interp(q,c,x[o]))

plt.close()
fig=plt.figure(figsize=(17*cm,13*cm))
gs0=fig.add_gridspec(nrows=2,hspace=0.55)
gs_top=gs0[0].subgridspec(1,2,wspace=0.45,width_ratios=[1,1.35])
gs_bot=gs0[1].subgridspec(1,2,wspace=0.5,width_ratios=[1,1.05])
model_handles=[Line2D([],[],c=cols[m],lw=1.5,label=models[m]) for m in range(3)]

#a: cumulative area below a given relative uncertainty, large ensemble vs. bootstrapped members (as plot_04_Fig3_uncertainty_skill.py)
ax=fig.add_subplot(gs_top[0])
for m, model in enumerate(models):
	chi_LE=xr.open_dataset(skill+"bias_maps_"+model+"_"+tag+".nc").chi_LE
	chi_BS=xr.open_dataset(skill+"chi_bootstrap_"+model+"_"+tag+".nc").chi
	w=np.cos(np.deg2rad(chi_LE.lat))
	af_LE=[float(w.broadcast_like(chi_LE).where(chi_LE<r,0).sum())/float(w.broadcast_like(chi_LE).sum()) for r in relerrs]
	af_BS=np.array([[float((chi_BS[n]<r).weighted(w).mean()) for r in relerrs] for n in range(chi_BS.member.size)])
	ax.plot(relerrs,af_LE,c=cols[m],lw=1.5)
	ax.plot(relerrs,np.median(af_BS,axis=0),c=cols[m],ls="--",lw=1.5)
	ax.fill_between(relerrs,np.percentile(af_BS,5,axis=0),np.percentile(af_BS,95,axis=0),color=cols[m],alpha=0.15,lw=0)
handles=model_handles+[Line2D([],[],c="grey",lw=1.5,label="Large ensemble"),Line2D([],[],c="grey",lw=1.5,ls="--",label="Bootstrap\n(median, 5-95%)")]
ax.legend(handles=handles,fontsize=fs-1,loc="upper left",frameon=False)
ax.set_xlabel("Relative uncertainty due to internal\n"+r"variability, $\chi$ = X (%)",fontsize=fs)
ax.set_ylabel("Fraction of global surface area\n"+r"with $\chi$ below X",fontsize=fs)
ax.set_xlim([relerrs[0],relerrs[-1]])
ax.set_ylim([0,1])
style(ax)
annotate_panel(ax,"a")

#b: pattern correlation of bootstrapped sigma (filled) and chi (open) maps with the large ensemble, median and 5-95% across members
ax=fig.add_subplot(gs_top[1])
for i, (lab, t, col) in enumerate(metrics):
	d=pd.read_csv(skill+"sigma_chi_members_"+t+".csv")
	for m, model in enumerate(models):
		dm=d[d.model==model]
		x=i+(m-1)*0.25
		for var, dx, face in [("r_sigma",-0.06,cols[m]),("r_chi",0.06,"white")]:
			ax.plot([x+dx]*2,dm[var].quantile([0.05,0.95]),c=cols[m],lw=0.8)
			ax.scatter(x+dx,dm[var].median(),s=14,facecolor=face,edgecolor=cols[m],lw=0.8,zorder=3)
	if i>0:
		ax.axvline(i-0.5,c="grey",lw=0.5)
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels([m[0] for m in metrics])
#tick labels in the metric colours of panel d
for tl, (lab, t, col) in zip(ax.get_xticklabels(),metrics):
	tl.set_color(col)
ax.set_xlim([-0.5,len(metrics)-0.5])
ax.set_ylim([0,1])
ax.set_ylabel("Weighted pattern correlation\nwith large ensemble",fontsize=fs)
handles=[Line2D([],[],ls="",marker="o",ms=4,color="grey",label=r"Absolute uncertainty ($\sigma$)"),
	Line2D([],[],ls="",marker="o",ms=4,markerfacecolor="white",color="grey",label=r"Relative uncertainty ($\chi$)")]
ax.legend(handles=handles,fontsize=fs-1,loc="lower center",bbox_to_anchor=(0.5,1.0),frameon=False,ncol=2)
style(ax)
ax.grid(False,axis="x")
annotate_panel(ax,"b",x=-0.15)

#c: median pattern correlation of bootstrapped sigma maps with each model's large ensemble (common CanESM5 grid)
ax=fig.add_subplot(gs_bot[0])
cross=pd.read_csv(skill+"cross_model_sigma_common_grid_"+tag+".csv")
mat=cross.groupby(["model_bs","model_le"])["r"].median().unstack().loc[models,models]
sns.heatmap(mat,ax=ax,cmap="Blues",annot=True,fmt=".2f",annot_kws={"fontsize":fs},linewidths=1,linecolor="white",square=True,
	cbar_kws={"shrink":0.8},xticklabels=models,yticklabels=models)
ax.collections[0].colorbar.ax.tick_params(labelsize=fs)
ax.collections[0].colorbar.set_label("Median pattern correlation of\n"+r"absolute uncertainty ($\sigma$)",fontsize=fs)
for k in range(3):
	ax.add_patch(plt.Rectangle((k,k),1,1,fill=False,ec="k",lw=1))
ax.set_xlabel("Large ensemble (reference)",fontsize=fs)
ax.set_ylabel("Model of bootstrapped member",fontsize=fs)
ax.tick_params(axis='both', which='major', labelsize=fs,length=0)
plt.setp(ax.get_xticklabels(),rotation=20,ha="right")
plt.setp(ax.get_yticklabels(),rotation=0)
ax.annotate("c",xy=(-0.55,1.06),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")
own=cross[cross.model_bs==cross.model_le]
print("panel c: sigma maps closest to own model: "+str(int(own.best_r.sum()))+"/"+str(len(own)))
print(mat.round(3))

#d: bootstrap against large-ensemble relative uncertainty, binned by the large-ensemble value of each grid cell (as Fig. A17b);
#line: median, shading: interquartile range across grid cells, members and models (area weighted, each model weighted equally);
#only bins covering at least 1% of the global area on average across models
bins=[0,20,40,70,100,150,200,300,500,1000,np.inf]
ax=fig.add_subplot(gs_bot[1])
for lab, t, col in metrics:
	chiLE=[];chiBS=[];wts=[]
	afrac=np.zeros(len(bins)-1)
	for model in models:
		cle=xr.open_dataset(skill+"bias_maps_"+model+"_"+t+".nc").chi_LE
		cbs=xr.open_dataset(skill+"chi_bootstrap_"+model+"_"+t+".nc").chi.transpose("member","lat","lon")
		w=np.cos(np.deg2rad(cle.lat)).broadcast_like(cle).values
		w=w/w.sum()
		afrac+=np.histogram(cle.values.ravel(),bins=bins,weights=w.ravel())[0]/len(models)
		nmem=cbs.member.size
		chiLE.append(np.tile(cle.values.ravel(),nmem))
		chiBS.append(cbs.values.ravel())
		wts.append(np.tile(w.ravel(),nmem)/nmem)
	chiLE=np.concatenate(chiLE);chiBS=np.concatenate(chiBS);wts=np.concatenate(wts)
	ok=np.isfinite(chiLE)&np.isfinite(chiBS)
	chiLE,chiBS,wts=chiLE[ok],chiBS[ok],wts[ok]
	x=[];med=[];p25=[];p75=[]
	for b in range(len(bins)-1):
		sel=(chiLE>=bins[b])&(chiLE<bins[b+1])
		if afrac[b]<0.01 or sel.sum()==0:
			continue
		x.append(wquantile(chiLE[sel],wts[sel],0.5))
		med.append(wquantile(chiBS[sel],wts[sel],0.5))
		p25.append(wquantile(chiBS[sel],wts[sel],0.25))
		p75.append(wquantile(chiBS[sel],wts[sel],0.75))
	ax.plot(x,med,c=col,lw=1.5,marker="o",ms=3,label=lab.replace("\n"," "))
	ax.fill_between(x,p25,p75,color=col,alpha=0.15,lw=0)
lim=[10,3000]
ax.plot(lim,lim,c="k",lw=0.8,ls="--",label="1:1")
#median of 100/|t| for t ~ N(0,1): relative uncertainty of a single realisation without forced signal
ax.axhline(100/0.6745,c="grey",lw=0.8,ls=":",label="No forced signal")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel(r"Relative uncertainty $\chi$, large ensemble (%)",fontsize=fs)
ax.set_ylabel(r"Relative uncertainty $\chi$, bootstrap (%)",fontsize=fs)
ax.legend(fontsize=fs-1,loc="upper left",frameon=False)
style(ax)
annotate_panel(ax,"d",x=-0.2)

name="figs/member_stats/Fig5_precip_sigma_chi_"+tag+".png"
plt.savefig(name,bbox_inches="tight",dpi=300)
plt.close()
print("saved "+name)
