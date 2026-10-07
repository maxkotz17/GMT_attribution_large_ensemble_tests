import numpy as np
import pandas as pd
import xarray as xr
import sys
import glob

#Absolute (sigma) vs relative (chi) uncertainty skill of the block-bootstrap against the large ensemble, and coverage of
#bootstrap confidence intervals. Outputs per metric:
#(a) per-member weighted pattern correlations of bootstrap sigma and chi maps with the large ensemble
#(b) bootstrap chi binned by large-ensemble chi (saturation of single-realisation chi at low signal-to-noise)
#(c) coverage: fraction of members whose 5-95% bootstrap interval of beta contains the ensemble-mean beta, by chi bin

varn=sys.argv[1] #tas / pr
metric=sys.argv[2] #0.99_exp_agg_NOPE_membspec or X5
W=int(sys.argv[3]) #5
relerr_cutoff=int(sys.argv[4]) #cap on chi for pattern correlations, 100 tas / 200 pr as in Figs 3/5
endyr="_"+sys.argv[5] if len(sys.argv)>5 else ""

models=["MIROC6","MPI-ESM1-2-LR","CanESM5"]
folder="/gpfs/scratch/bsc32/bsc400019/"
outfolder=folder+"attribution/skill/"
tag=varn+"_"+metric+"_block"+str(W)+"_"+str(relerr_cutoff)+endyr
#bins of large-ensemble relative uncertainty (%)
bins=[0,20,40,70,100,150,200,300,500,1000,np.inf]

#file conventions as in plot_04_Fig3_uncertainty_ag.py
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

def wcorr(a,b,w):
	m=np.isfinite(a)&np.isfinite(b)
	a,b,w=a[m],b[m],w[m]
	a=a-np.average(a,weights=w)
	b=b-np.average(b,weights=w)
	return(np.average(a*b,weights=w)/np.sqrt(np.average(a*a,weights=w)*np.average(b*b,weights=w)))

def wquantile(x,w,q):
	m=np.isfinite(x)
	x,w=x[m],w[m]
	o=np.argsort(x)
	c=np.cumsum(w[o])/w.sum()
	return(np.interp(q,c,x[o]))

rows_r=[]
rows_bin=[]
for model in models:
	coefs,coefs_bs=load_coefs(model)
	nmem=coefs_bs.member.size
	#large ensemble: mean (forced response), spread and relative uncertainty of beta
	mu_LE=coefs.mean("member").values
	sd_LE=coefs.std("member").values
	chi_LE=100*sd_LE/np.abs(mu_LE)
	#bootstrap per member (failed fits stored as NaN are skipped)
	mu_BS=coefs_bs.mean("sample",skipna=True).values
	sd_BS=coefs_bs.std("sample",skipna=True).values
	chi_BS=100*sd_BS/np.abs(mu_BS)
	lo=np.nanpercentile(coefs_bs.values,5,axis=0)
	hi=np.nanpercentile(coefs_bs.values,95,axis=0)
	covered=(lo<=mu_LE[None])&(mu_LE[None]<=hi)
	w=np.broadcast_to(np.cos(np.deg2rad(coefs.lat.values))[:,None],mu_LE.shape).ravel()

	#(a) pattern correlations of sigma and (capped) chi, per member
	for n in range(nmem):
		rows_r.append({"model":model,"member":n,
			"r_sigma":wcorr(sd_BS[n].ravel(),sd_LE.ravel(),w),
			"r_chi":wcorr(np.minimum(chi_BS[n],relerr_cutoff).ravel(),np.minimum(chi_LE,relerr_cutoff).ravel(),w),
			"coverage":np.average(covered[n].ravel(),weights=w)})

	#(b,c) by bin of large-ensemble chi: area-weighted median and quartiles of bootstrap chi (all members pooled), and coverage
	chiLE_all=np.broadcast_to(chi_LE,chi_BS.shape).ravel()
	w_all=np.tile(w,nmem)
	chiBS_all=chi_BS.ravel()
	cov_all=covered.ravel()
	for b in range(len(bins)-1):
		sel=(chiLE_all>=bins[b])&(chiLE_all<bins[b+1])
		if sel.sum()==0:
			continue
		rows_bin.append({"model":model,"bin_lo":bins[b],"bin_hi":bins[b+1],
			"area_frac":float(w[(chi_LE.ravel()>=bins[b])&(chi_LE.ravel()<bins[b+1])].sum()/w.sum()),
			"chiLE_median":wquantile(chiLE_all[sel],w_all[sel],0.5),
			"chiBS_p25":wquantile(chiBS_all[sel],w_all[sel],0.25),
			"chiBS_median":wquantile(chiBS_all[sel],w_all[sel],0.5),
			"chiBS_p75":wquantile(chiBS_all[sel],w_all[sel],0.75),
			"coverage":np.average(cov_all[sel],weights=w_all[sel])})
	print(model + " done: median r_sigma %.2f, r_chi %.2f, coverage of 90%% intervals %.3f"%(
		np.median([r["r_sigma"] for r in rows_r if r["model"]==model]),np.median([r["r_chi"] for r in rows_r if r["model"]==model]),
		np.median([r["coverage"] for r in rows_r if r["model"]==model])))
	del coefs_bs

pd.DataFrame(rows_r).to_csv(outfolder+"sigma_chi_members_"+tag+".csv",index=False)
bins_df=pd.DataFrame(rows_bin)
bins_df.to_csv(outfolder+"chi_bins_coverage_"+tag+".csv",index=False)
pd.set_option("display.width",200)
print(bins_df.round(3).to_string())
