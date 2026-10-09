import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#Figure A18: practical inference from the block-bootstrap of a single realisation, taking the large ensemble as ground truth
#a: actual against nominal coverage of bootstrap percentile intervals of the linear-regression slope, i.e. the fraction of
#   grid-cells and members whose interval contains the forced response (ensemble-mean slope)
#b: by bin of bootstrapped relative uncertainty, the fraction of grid-cells with large-ensemble relative uncertainty below 100%
#c: by bin of bootstrapped relative uncertainty, the fraction of grid-cells in which the sign of the bootstrapped slope
#   matches that of the forced response
#area-weighted and pooled across members; lines: mean across models, shading: range across models
#inputs from 20_bootstrap_ci_reliability.py

skill="/gpfs/scratch/bsc32/bsc400019/attribution/skill/"
#metric label, file tag, colour (as Figure 5)
metrics=[(r"$T5d$","tas_X5_block5_100","#b2182b"),
	(r"$T99p$","tas_0.99_exp_agg_NOPE_membspec_block5_100","#ef8a62"),
	(r"$P5d$","pr_X5_block5_200_2025","#2166ac"),
	(r"$P99p$","pr_0.99_exp_agg_NOPE_membspec_block5_200","#67a9cf"),
	(r"$P5d$ (1940-2100)","pr_X5_block5_200_2100","#542788")]
min_area=0.005 #only bins covering at least 0.5% of the global area, in at least two models

fs=6
cm=1/2.54

def style(ax):
	ax.tick_params(axis='both', which='major', labelsize=fs)
	ax.grid(which="major",axis="both",lw=0.5,alpha=0.5)
	ax.set_axisbelow(True)

def annotate_panel(ax,letter,x=-0.2):
	ax.annotate(letter,xy=(x,1.04),xycoords="axes fraction",fontsize=fs+2,fontweight="bold")

#mean and range across models for each bin
def across_models(d,var):
	d=d[d.area>=min_area]
	g=d.groupby("bin",sort=False)
	out=pd.DataFrame({"x":g.x.mean(),"mean":g[var].mean(),"lo":g[var].min(),"hi":g[var].max(),"n":g[var].size()})
	return(out[out.n>=2].sort_values("x"))

plt.close()
fig=plt.figure(figsize=(18*cm,6.5*cm))
gs=fig.add_gridspec(ncols=3,nrows=1,wspace=0.5)
axs=[fig.add_subplot(gs[i]) for i in range(3)]
for lab, t, col in metrics:
	d=pd.read_csv(skill+"ci_reliability_"+t+".csv")
	lev=d[d.kind=="level"].copy()
	lev["x"]=lev.x.astype(float)
	panels=[(axs[0],lev,"coverage"),(axs[1],d[d.kind=="chiBS"],"chiLE_lt100"),(axs[2],d[d.kind=="chiBS"],"sign_ok")]
	for ax, dd, var in panels:
		a=across_models(dd,var)
		ax.plot(a.x,a["mean"]*100,c=col,lw=1.2,marker="o",ms=2.5)
		ax.fill_between(a.x,a.lo*100,a.hi*100,color=col,alpha=0.15,lw=0)

axs[0].plot([45,100],[45,100],c="k",lw=0.8,ls="--")
axs[0].set_xlim([45,100])
axs[0].set_ylim([45,100])
axs[0].set_xlabel("Nominal confidence level of\nbootstrap interval (%)",fontsize=fs)
axs[0].set_ylabel("Bootstrap intervals containing\nthe forced response (%)",fontsize=fs)
axs[0].legend(handles=[Line2D([],[],c=col,lw=1.2,label=lab) for lab, t, col in metrics]+[Line2D([],[],c="k",lw=0.8,ls="--",label="1:1")],
	fontsize=fs-1,loc="upper left",frameon=False)
axs[1].set_ylabel("Grid-cells with large-ensemble\n"+r"$\chi$ below 100% (%)",fontsize=fs)
axs[1].set_ylim([0,100])
axs[2].set_ylabel("Grid-cells with sign of forced\nresponse correctly identified (%)",fontsize=fs)
axs[2].set_ylim([0,100])
for ax in axs[1:]:
	ax.set_xscale("log")
	ax.set_xlim([10,1000])
	ax.set_xlabel("Relative uncertainty from\n"+r"bootstrap, $\chi$ (%)",fontsize=fs)
for ax, letter in zip(axs,"abc"):
	style(ax)
	annotate_panel(ax,letter)

name="figs/member_stats/FigA18_bootstrap_inference.png"
plt.savefig(name,bbox_inches="tight",dpi=300)
plt.close()
print("saved "+name)
