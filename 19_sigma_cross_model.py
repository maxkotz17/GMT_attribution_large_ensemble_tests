import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import sys
import glob

#Cross-model benchmark for ABSOLUTE uncertainty (sigma, standard deviation of the GMT regression coefficient): is the
#bootstrapped sigma map of each member closer to the large-ensemble sigma map of its own model than to those of the other
#models? As (1b) in 17_bootstrap_skill_stats.py (all maps conservatively regridded to the common CanESM5 grid), but for
#sigma instead of capped chi. sigma is uncapped, as in the pattern correlations of 18_sigma_chi_coverage.py

varn=sys.argv[1] #tas / pr
metric=sys.argv[2] #0.99_exp_agg_NOPE_membspec or X5
W=int(sys.argv[3]) #5
relerr_cutoff=int(sys.argv[4]) #only used in the output tag, to match the other skill files (100 tas / 200 pr)
endyr="_"+sys.argv[5] if len(sys.argv)>5 else ""

models=["MIROC6","MPI-ESM1-2-LR","CanESM5"]
folder="/gpfs/scratch/bsc32/bsc400019/"
outfolder=folder+"attribution/skill/"
tag=varn+"_"+metric+"_block"+str(W)+"_"+str(relerr_cutoff)+endyr

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

def with_bounds(da):
	lat=da.lat.values
	lon=da.lon.values
	lat_b=np.concatenate(([-90],(lat[1:]+lat[:-1])/2,[90]))
	dlon=np.mean(np.diff(lon))
	lon_b=np.concatenate(([lon[0]-dlon/2],(lon[1:]+lon[:-1])/2,[lon[-1]+dlon/2]))
	return(xr.Dataset(coords={"lat":lat,"lon":lon,"lat_b":lat_b,"lon_b":lon_b}))

#weighted pattern correlation of vector a with each row of B, skipping non-finite cells (failed Poisson fits)
def wcorr_rows(a,B,w):
	out=[]
	for b in B:
		m=np.isfinite(a)&np.isfinite(b)
		x,y,ww=a[m],b[m],w[m]
		x=x-np.average(x,weights=ww)
		y=y-np.average(y,weights=ww)
		out.append(np.average(x*y,weights=ww)/np.sqrt(np.average(x*x,weights=ww)*np.average(y*y,weights=ww)))
	return(np.array(out))

common={}
target=None
for model in ["CanESM5","MIROC6","MPI-ESM1-2-LR"]:
	coefs,coefs_bs=load_coefs(model)
	sd_LE=coefs.std("member")
	sd_BS=coefs_bs.std("sample",skipna=True)
	del coefs_bs
	if model=="CanESM5":
		target=with_bounds(sd_LE)
		common[model]={"LE":sd_LE,"BS":sd_BS}
	else:
		regridder=xe.Regridder(with_bounds(sd_LE),target,"conservative",periodic=True)
		common[model]={"LE":regridder(sd_LE),"BS":regridder(sd_BS)}
	print(model+" loaded")

w=np.repeat(np.cos(np.deg2rad(common["CanESM5"]["LE"].lat.values)),common["CanESM5"]["LE"].lon.size)
LEmaps=np.array([common[model]["LE"].values.flatten() for model in models])
rows=[]
for model_bs in models:
	A=common[model_bs]["BS"].transpose("member","lat","lon").values
	A=A.reshape(A.shape[0],-1)
	for n in range(A.shape[0]):
		R=wcorr_rows(A[n],LEmaps,w)
		for j, model_le in enumerate(models):
			rows.append({"model_bs":model_bs,"member":n,"model_le":model_le,"r":R[j],"best_r":models[np.argmax(R)]==model_le})
rows=pd.DataFrame(rows)
rows.to_csv(outfolder+"cross_model_sigma_common_grid_"+tag+".csv",index=False)

own=rows[rows.model_bs==rows.model_le]
print(tag+": sigma maps closest to own model (common CanESM5 grid): "+str(int(own.best_r.sum()))+"/"+str(len(own)))
print(rows.groupby(["model_bs","model_le"]).r.median().unstack().loc[models,models].round(3))
#large ensembles against each other, for reference
print("LE vs LE sigma correlations:")
print(pd.DataFrame([wcorr_rows(LEmaps[i],LEmaps,w) for i in range(3)],index=models,columns=models).round(3))
