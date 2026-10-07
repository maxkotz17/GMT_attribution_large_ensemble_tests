import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import cartopy.crs as ccrs
import sys

#Figures for the review: skill of the block-bootstrapped relative uncertainty against the large ensembles
#inputs from 17_bootstrap_skill_stats.py

varn=sys.argv[1] #tas
metric=sys.argv[2] #0.99_exp_agg_NOPE_membspec or X5
W=int(sys.argv[3]) #5
relerr_cutoff=int(sys.argv[4]) #100
#optional end year suffix of the coefficient files (e.g. 2025 for pr X5), as in plot_04
endyr="_"+sys.argv[5] if len(sys.argv)>5 else ""

models=["MIROC6","MPI-ESM1-2-LR","CanESM5"]
labs=["MIROC6","MPI-ESM1-2-LR","CanESM5"]
cols=["tab:blue","tab:orange","tab:green"]
folder="/gpfs/scratch/bsc32/bsc400019/attribution/skill/"
tag=varn+"_"+metric+"_block"+str(W)+"_"+str(relerr_cutoff)+endyr

cross=pd.read_csv(folder+"cross_model_"+tag+".csv")
obs=pd.read_csv(folder+"obs_skill_"+tag+".csv")
thr=pd.read_csv(folder+"threshold_skill_"+tag+".csv")
nc=pd.concat([pd.read_csv(folder+"noise_ceiling_"+model+"_"+tag+".csv",index_col=0) for model in models])
r_boot={model:np.load(folder+"r_boot_pairs_"+model+"_"+tag+".npy") for model in models}

fs=6
cm=1/2.54
#legends go where the data are not: correlations are high for temperature, low for precipitation
leg_low=varn=="tas"
#only show thresholds where at least 1% of the area is truly below them in the large ensemble (else precision/hit rate are noise)
min_base=0.01
w1=5.5*cm
h1=5*cm

def annotate_panel(ax,letter):
	ax.annotate(letter,xy=(-0.18,1.05),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

def style(ax):
	ax.tick_params(axis='both', which='major', labelsize=fs)
	ax.grid(which="major",axis="both",lw=0.5,alpha=0.5)
	ax.set_axisbelow(True)

#FIGURE 1: skill summary
plt.close()
fig=plt.figure(figsize=(3*w1+1.2,2*h1+0.9))
gs=fig.add_gridspec(ncols=3,nrows=2,wspace=0.55,hspace=0.6)

#a,b: cross-model matrices (rows: model of the bootstrapped member, columns: large ensemble used as reference)
for p, (var, cmap, lab, fmt) in enumerate([("r","Blues","Median pattern correlation",".2f"),("rmse","Blues_r","Median weighted RMSE (%-points)",".0f")]):
	ax=fig.add_subplot(gs[0,p])
	mat=cross.groupby(["model_bs","model_le"])[var].median().unstack().loc[models,models]
	sns.heatmap(mat,ax=ax,cmap=cmap,annot=True,fmt=fmt,annot_kws={"fontsize":fs},linewidths=1,linecolor="white",square=True,
		cbar_kws={"shrink":0.8},xticklabels=labs,yticklabels=labs)
	ax.collections[0].colorbar.ax.tick_params(labelsize=fs)
	ax.collections[0].colorbar.set_label(lab,fontsize=fs)
	for k in range(3):
		ax.add_patch(plt.Rectangle((k,k),1,1,fill=False,ec="k",lw=1))
	ax.set_xlabel("Large ensemble (reference)",fontsize=fs)
	if p==0:
		ax.set_ylabel("Model of bootstrapped member",fontsize=fs)
	else:
		ax.set_ylabel("")
		ax.set_yticklabels([])
	ax.tick_params(axis='both', which='major', labelsize=fs,length=0)
	plt.setp(ax.get_xticklabels(),rotation=20,ha="right")
	plt.setp(ax.get_yticklabels(),rotation=0)
	key="best_r" if var=="r" else "best_rmse"
	own=cross[cross.model_bs==cross.model_le]
	ax.set_title(str(int(own[key].sum()))+"/"+str(len(own))+" members matched to own model",fontsize=fs,pad=8)
	ax.annotate(["a","b"][p],xy=(-0.12 if p else -0.5,1.2),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

#c: own large ensemble vs best-matching other large ensemble, per member
ax=fig.add_subplot(gs[0,2])
own=cross[cross.model_bs==cross.model_le].set_index(["model_bs","member"]).r
other=cross[cross.model_bs!=cross.model_le].groupby(["model_bs","member"]).r.max()
for m, model in enumerate(models):
	o=own.loc[model].values
	b=other.loc[model].values
	for k in range(len(o)):
		ax.plot([m-0.15,m+0.15],[b[k],o[k]],c=cols[m],lw=0.4,alpha=0.4)
	ax.scatter(np.full(len(b),m-0.15),b,s=8,facecolor="white",edgecolor=cols[m],lw=0.6,zorder=3)
	ax.scatter(np.full(len(o),m+0.15),o,s=8,color=cols[m],edgecolor="white",lw=0.3,zorder=3)
ax.scatter([],[],s=8,facecolor="white",edgecolor="grey",label="Best other large ensemble")
ax.scatter([],[],s=8,color="grey",label="Own large ensemble")
ax.set_xticks(range(3))
ax.set_xticklabels(labs,rotation=20,ha="right")
ax.set_ylim([0,1])
ax.set_ylabel("Weighted pattern correlation",fontsize=fs)
ax.legend(fontsize=fs,loc="lower right" if leg_low else "upper left",frameon=False)
style(ax)
annotate_panel(ax,"c")

#d: noise ceiling
ax=fig.add_subplot(gs[1,0])
for m, model in enumerate(models):
	rb=r_boot[model]
	r_obs=obs[obs.model==model].r_obs_LOO.values
	ax.plot([m-0.25]*2,np.percentile(rb,[5,95]),c=cols[m],lw=1)
	ax.scatter(m-0.25,np.median(rb),marker="s",s=12,color=cols[m],edgecolor="white",lw=0.3,zorder=3)
	ax.scatter(m-0.25,nc.loc[model,"r_LE50"],marker="^",s=14,color=cols[m],edgecolor="white",lw=0.3,zorder=3)
	ax.plot([m-0.05,m+0.35],[nc.loc[model,"ceiling"]]*2,c="k",lw=1)
	parts=ax.violinplot(r_obs,positions=[m+0.15],widths=0.3,showextrema=False,showmedians=True)
	for pc in parts["bodies"]:
		pc.set_facecolor(cols[m])
		pc.set_alpha(0.4)
	parts["cmedians"].set_color(cols[m])
ax.scatter([],[],marker="^",s=14,color="grey",label="Large ensemble reliability")
ax.scatter([],[],marker="s",s=12,color="grey",label="Bootstrap reliability (5-95%)")
ax.plot([],[],c="k",lw=1,label="Expected if bootstrap unbiased")
ax.fill_between([],[],[],color="grey",alpha=0.4,label="Observed (leave-one-out)")
ax.set_xticks(range(3))
ax.set_xticklabels(labs,rotation=20,ha="right")
ax.set_ylim([0,1])
ax.set_ylabel("Weighted pattern correlation",fontsize=fs)
if leg_low:
	ax.legend(fontsize=fs-1,loc="lower left",frameon=False)
else:
	ax.legend(fontsize=fs-1,loc="center",bbox_to_anchor=(0.5,0.6),frameon=False)
style(ax)
annotate_panel(ax,"d")

#e,f: threshold precision and hit rate, median and 5-95% across members, with the no-skill reference (area truly below threshold)
for p, (var, lab) in enumerate([("precision",r"Precision (classed $\chi$<X and truly $\chi$<X)"),("hit_rate",r"Hit rate (truly $\chi$<X and classed $\chi$<X)")]):
	ax=fig.add_subplot(gs[1,p+1])
	for m, model in enumerate(models):
		dm=thr[thr.model==model]
		keep=dm.groupby("thresh").base_rate.median()>=min_base
		d=dm[dm.thresh.isin(keep.index[keep])].groupby("thresh")
		x=np.array(list(d.groups.keys()))
		ax.plot(x,d[var].median(),c=cols[m],lw=1.5,marker="o",ms=3,label=model)
		ax.fill_between(x,d[var].quantile(0.05),d[var].quantile(0.95),color=cols[m],alpha=0.15,lw=0)
		if var=="precision":
			ax.plot(x,d["base_rate"].median(),c=cols[m],lw=1,ls=":")
	if var=="precision":
		ax.plot([],[],c="grey",lw=1,ls=":",label="No skill (area truly $\\chi$<X)")
	ax.set_xticks(np.sort(thr.thresh.unique()))
	ax.set_xlabel("Relative uncertainty threshold X (%)",fontsize=fs)
	ax.set_ylabel(lab,fontsize=fs)
	ax.set_ylim([0,1])
	ax.legend(fontsize=fs-1,loc="lower right" if leg_low else "upper left",frameon=False)
	style(ax)
	annotate_panel(ax,["e","f"][p])

plt.savefig("figs/boot_stats/review_skill_" + tag + ".png",bbox_inches="tight",dpi=300)
plt.close()

#FIGURE 2: bias maps, median across members of the bootstrap / large-ensemble ratio (leave-one-out LE)
#log chi ratio = log sigma ratio - log |mu| ratio
fields=[("logratio_chi",r"$\hat\chi$ / $\chi_{LE}$"),("logratio_std",r"$\hat\sigma$ / $\sigma_{LE}$"),("logratio_absmu",r"$|\hat\mu|$ / $|\mu_{LE}|$")]
ticks=[0.5,0.67,0.8,1,1.25,1.5,2]
norm=mcolors.Normalize(vmin=np.log(0.5),vmax=np.log(2))
projection=ccrs.Robinson()
fig=plt.figure(figsize=(3*6*cm,3*3.4*cm+1))
gs=fig.add_gridspec(ncols=3,nrows=4,height_ratios=[1,1,1,0.07],wspace=0.05,hspace=0.3)
for m, model in enumerate(models):
	bias=xr.open_dataset(folder+"bias_maps_"+model+"_"+tag+".nc")
	wgt=np.cos(np.deg2rad(bias.lat))
	for f, (field, lab) in enumerate(fields):
		ax=fig.add_subplot(gs[m,f],projection=projection)
		pl=ax.pcolormesh(bias.lon,bias.lat,bias[field],transform=ccrs.PlateCarree(),cmap="RdBu_r",norm=norm,rasterized=True)
		ax.coastlines(lw=0.4)
		#large-ensemble relative uncertainty of 100%: low signal-to-noise beyond this contour
		ax.contour(bias.lon,bias.lat,bias["chi_LE"],levels=[100],colors="k",linewidths=0.6,linestyles="--",transform=ccrs.PlateCarree())
		ax.set_global()
		av=float(np.exp(bias[field].weighted(wgt).mean()))
		ax.text(0.5,-0.03,"area-weighted mean ratio " + "{:.2f}".format(av),transform=ax.transAxes,ha="center",va="top",fontsize=fs)
		if m==0:
			ax.set_title(lab,fontsize=fs+2,pad=8)
		if f==0:
			ax.text(-0.04,0.5,model,transform=ax.transAxes,rotation=90,va="center",ha="right",fontsize=fs+1)
		ax.annotate("abcdefghi"[3*m+f],xy=(0,1.06),xycoords="axes fraction",fontsize=fs+2,fontweight="bold",ha="right")
cax=fig.add_subplot(gs[3,:])
cbar=fig.colorbar(pl,cax=cax,orientation="horizontal",extend="both",ticks=np.log(ticks))
cbar.ax.set_xticklabels([str(t) for t in ticks])
cbar.ax.tick_params(labelsize=fs)
cbar.set_label("Median ratio bootstrap / large ensemble across members (log scale); dashed: large-ensemble "+r"$\chi$=100%",fontsize=fs)
pos=cax.get_position()
cax.set_position([pos.x0+0.2*pos.width,pos.y0,0.6*pos.width,pos.height])
plt.savefig("figs/boot_stats/review_bias_maps_" + tag + ".png",bbox_inches="tight",dpi=300)
plt.close()

print(nc.round(3))
