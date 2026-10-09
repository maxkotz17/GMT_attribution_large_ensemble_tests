import numpy as np
import pandas as pd
import xarray as xr
import sys
import glob

#Reliability of single-realisation block-bootstrap inference against the large ensemble (taken as ground truth):
#(1) coverage: fraction of grid-cells/members whose bootstrap percentile interval of beta contains the ensemble-mean beta,
#    for several nominal levels, and for the 90% interval by bin of large-ensemble and of bootstrapped relative uncertainty
#(2) detection: by bin of bootstrapped relative uncertainty, the fraction of grid-cells/members in which the sign of the
#    bootstrapped beta matches the sign of the ensemble-mean beta, and in which the large-ensemble chi is below 100%
#all statistics area-weighted and pooled across members, separately for each model

varn=sys.argv[1] #tas / pr
metric=sys.argv[2] #0.99_exp_agg_NOPE_membspec or X5
W=int(sys.argv[3]) #5
relerr_cutoff=int(sys.argv[4]) #only used in the output tag (100 tas / 200 pr)
endyr="_"+sys.argv[5] if len(sys.argv)>5 else ""

models=["MIROC6","MPI-ESM1-2-LR","CanESM5"]
folder="/gpfs/scratch/bsc32/bsc400019/"
outfolder=folder+"attribution/skill/"
tag=varn+"_"+metric+"_block"+str(W)+"_"+str(relerr_cutoff)+endyr
levels=[50,68,80,90,95]
bins=[0,20,30,40,50,60,80,100,150,200,300,500,np.inf]

#file conventions as in 17_bootstrap_skill_stats.py
def load_coefs(model):
	if "0.99" in metric:
		coefs=xr.concat([xr.open_dataset(folder + "attribution/attr_coefs_" + model + "_" + varn + "_" + metric + "_" + str(i) + ".nc").coef for i in range(50)],dim="member")
		files=np.sort(glob.glob(folder + "attribution/bootstrap/attr_coefs_" + model + "_" + varn + "_" + metric + "_block"+str(W)+"_seed*.nc"))
		coefs_bs=xr.concat([xr.open_dataset(f).coef for f in files],dim="sample")
	else:
		coefs=xr.open_dataset(folder + "attribution/attr_coefs_" + model + "_" + varn + "_" + metric + endyr + ".nc").coef
		coefs_bs=xr.open_dataset(folder + "attribution/bootstrap/attr_coefs_" + model + "_" + varn + "_" + metric + "_block"+str(W) + endyr + ".nc").coef
	coefs=coefs.sortby("lon").sortby("lat").transpose("member","lat","lon").load()
	coefs_bs=coefs_bs.sortby("lon").sortby("lat").transpose("sample","member","lat","lon").load()
	assert np.allclose(coefs.lat,coefs_bs.lat) and np.allclose(coefs.lon,coefs_bs.lon), "grids differ"
	coefs_bs=coefs_bs.assign_coords(lat=coefs.lat,lon=coefs.lon)
	coefs=coefs.sel(member=coefs_bs.member.values)
	return(coefs,coefs_bs)

rows=[]
for model in models:
	coefs,coefs_bs=load_coefs(model)
	mu_LE=coefs.mean("member").values
	chi_LE=100*coefs.std("member").values/np.abs(mu_LE)
	bs=coefs_bs.values
	av=np.nanmean(bs,axis=0)
	chi_BS=100*np.nanstd(bs,axis=0,ddof=1)/np.abs(av)
	nmem=av.shape[0]
	w=np.cos(np.deg2rad(coefs.lat.values))[:,None]*np.ones(mu_LE.shape)
	L=np.broadcast_to(chi_LE,av.shape).ravel()
	B=chi_BS.ravel()
	Wt=np.broadcast_to(w,av.shape).ravel()
	sign_ok=(np.sign(av)==np.sign(mu_LE)[None]).ravel()
	#percentile intervals for all levels in one call; nanpercentile (slow) only for the few cells with failed Poisson fits
	qs=[q for lev in levels for q in (50-lev/2,50+lev/2)]
	P=np.percentile(bs,qs,axis=0)
	bad=np.isnan(bs).any(axis=0)
	if bad.any():
		P[:,bad]=np.nanpercentile(bs[:,bad],qs,axis=0)
	covered={}
	for i, lev in enumerate(levels):
		covered[lev]=((P[2*i]<=mu_LE[None])&(mu_LE[None]<=P[2*i+1])).ravel()
	del P
	del bs, coefs_bs
	ok=np.isfinite(L)&np.isfinite(B)
	#(1a) coverage at each nominal level, pooled, and its spread across members
	for lev in levels:
		cm=covered[lev].reshape(nmem,-1)
		okm=ok.reshape(nmem,-1)
		wm=Wt.reshape(nmem,-1)
		per_member=[np.average(cm[n][okm[n]],weights=wm[n][okm[n]]) for n in range(nmem)]
		rows.append({"model":model,"kind":"level","bin":lev,"x":lev,"area":1.0,"coverage":np.average(covered[lev][ok],weights=Wt[ok]),
			"coverage_p05":np.percentile(per_member,5),"coverage_p95":np.percentile(per_member,95)})
	#(1b,c,2) by bin of large-ensemble and of bootstrapped chi
	for kind, X in [("chiLE",L),("chiBS",B)]:
		for b in range(len(bins)-1):
			sel=ok&(X>=bins[b])&(X<bins[b+1])
			if Wt[sel].sum()==0:
				continue
			o=np.argsort(X[sel])
			c=np.cumsum(Wt[sel][o])/Wt[sel].sum()
			rows.append({"model":model,"kind":kind,"bin":"%g-%g"%(bins[b],bins[b+1]),"x":np.interp(0.5,c,X[sel][o]),
				"area":Wt[sel].sum()/Wt[ok].sum(),"coverage":np.average(covered[90][sel],weights=Wt[sel]),
				"sign_ok":np.average(sign_ok[sel],weights=Wt[sel]),"chiLE_lt100":np.average(L[sel]<100,weights=Wt[sel])})
	print(model+" done: coverage of 90%% intervals %.3f"%[r["coverage"] for r in rows if r["model"]==model and r["kind"]=="level" and r["bin"]==90][0],flush=True)

pd.DataFrame(rows).to_csv(outfolder+"ci_reliability_"+tag+".csv",index=False)
print("saved "+outfolder+"ci_reliability_"+tag+".csv")
