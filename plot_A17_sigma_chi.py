import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#Figure A17: the block-bootstrap reproduces absolute uncertainty (sigma) for both temperature and precipitation extremes,
#whereas single-realisation relative uncertainty (chi) saturates where the signal-to-noise ratio is low
#inputs from 18_sigma_chi_coverage.py

skill="/gpfs/scratch/bsc32/bsc400019/attribution/skill/"
models=["MIROC6","MPI-ESM1-2-LR","CanESM5"]
model_cols=["tab:blue","tab:orange","tab:green"]
#metric label, file tag, colour (warm: temperature, cool: precipitation)
metrics=[(r"$T5d$","tas_X5_block5_100","#b2182b"),
	(r"$T99p$","tas_0.99_exp_agg_NOPE_membspec_block5_100","#ef8a62"),
	(r"$P5d$","pr_X5_block5_200_2025","#2166ac"),
	(r"$P99p$","pr_0.99_exp_agg_NOPE_membspec_block5_200","#67a9cf"),
	(r"$P5d$"+"\n(1940-2100)","pr_X5_block5_200_2100","#542788")]

fs=6
cm=1/2.54
plt.close()
fig=plt.figure(figsize=(18*cm,7*cm))
gs=fig.add_gridspec(ncols=2,nrows=1,width_ratios=[1.3,1],wspace=0.3)

def style(ax):
	ax.tick_params(axis='both', which='major', labelsize=fs)
	ax.grid(which="major",axis="both",lw=0.5,alpha=0.5)
	ax.set_axisbelow(True)

#a: pattern correlation of bootstrap sigma (filled) and chi (open) maps with the large ensemble, median and 5-95% across members
ax=fig.add_subplot(gs[0])
for i, (lab, tag, col) in enumerate(metrics):
	d=pd.read_csv(skill+"sigma_chi_members_"+tag+".csv")
	for m, model in enumerate(models):
		dm=d[d.model==model]
		x=i+(m-1)*0.25
		for var, dx, face in [("r_sigma",-0.06,model_cols[m]),("r_chi",0.06,"white")]:
			ax.plot([x+dx]*2,dm[var].quantile([0.05,0.95]),c=model_cols[m],lw=0.8)
			ax.scatter(x+dx,dm[var].median(),s=14,facecolor=face,edgecolor=model_cols[m],lw=0.8,zorder=3)
	if i>0:
		ax.axvline(i-0.5,c="grey",lw=0.5)
ax.set_xticks(range(len(metrics)))
ax.set_xticklabels([m[0] for m in metrics])
ax.set_xlim([-0.5,len(metrics)-0.5])
ax.set_ylim([0,1])
ax.set_ylabel("Weighted pattern correlation between\nlarge ensemble and bootstrapped members",fontsize=fs)
handles=[Line2D([],[],ls="",marker="o",ms=4,color="grey",label=r"Absolute uncertainty ($\sigma$)"),
	Line2D([],[],ls="",marker="o",ms=4,markerfacecolor="white",color="grey",label=r"Relative uncertainty ($\chi$)")]
handles+=[Line2D([],[],c=model_cols[m],lw=1.5,label=models[m]) for m in range(3)]
ax.legend(handles=handles,fontsize=fs-1,loc="lower center",bbox_to_anchor=(0.5,1.01),frameon=False,ncol=3)
style(ax)
ax.grid(False,axis="x")
ax.annotate("a",xy=(-0.14,1.12),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

#b: bootstrap relative uncertainty against large-ensemble relative uncertainty, binned by the large-ensemble value of each
#grid cell; line: median, shading: interquartile range across grid cells, ensemble members and models (pooled, area weighted,
#each model weighted equally); only bins covering at least 1% of the global area on average across models
bins=[0,20,40,70,100,150,200,300,500,1000,np.inf]
def wquantile(x,w,q):
	o=np.argsort(x)
	c=np.cumsum(w[o])/w.sum()
	return(np.interp(q,c,x[o]))
ax=fig.add_subplot(gs[1])
for lab, tag, col in metrics:
	chiLE=[]
	chiBS=[]
	wts=[]
	afrac=np.zeros(len(bins)-1)
	for model in models:
		cle=xr.open_dataset(skill+"bias_maps_"+model+"_"+tag+".nc").chi_LE
		cbs=xr.open_dataset(skill+"chi_bootstrap_"+model+"_"+tag+".nc").chi.transpose("member","lat","lon")
		w=np.cos(np.deg2rad(cle.lat)).broadcast_like(cle).values
		w=w/w.sum()
		afrac+=np.histogram(cle.values.ravel(),bins=bins,weights=w.ravel())[0]/len(models)
		nmem=cbs.member.size
		chiLE.append(np.tile(cle.values.ravel(),nmem))
		chiBS.append(cbs.values.ravel())
		wts.append(np.tile(w.ravel(),nmem)/nmem)
	chiLE=np.concatenate(chiLE)
	chiBS=np.concatenate(chiBS)
	wts=np.concatenate(wts)
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
	print(tag,"bins:",np.round(x).astype(int).tolist(),"median chi_BS:",np.round(med).astype(int).tolist())
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
ax.annotate("b",xy=(-0.2,1.12),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

plt.savefig("figs/member_stats/FigA17_sigma_chi.png",bbox_inches="tight",dpi=300)
plt.close()
