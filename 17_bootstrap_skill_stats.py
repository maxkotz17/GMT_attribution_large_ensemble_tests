import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import sys
import glob
import os

#Skill statistics of the block-bootstrapped relative uncertainty (chi) against the large ensemble,
#for the review: (1) cross-model benchmark, (2) noise ceiling, (3) threshold precision/hit rate, (4) bias maps
#chi = 100*std/|mean| of the GMT regression coefficient, as in plot_04_Fig3_uncertainty_ag.py

varn=sys.argv[1] #tas
metric=sys.argv[2] #0.99_exp_agg_NOPE_membspec or X5
W=int(sys.argv[3]) #block width, 5
relerr_cutoff=int(sys.argv[4]) #cap on chi for pattern correlations / RMSE, 100 as in Fig. 3
#optional end year suffix of the coefficient files (e.g. 2025 for pr X5), as in plot_04
endyr="_"+sys.argv[5] if len(sys.argv)>5 else ""
nsplit=100 #random split-halves of the large ensemble for the noise ceiling

models=["MIROC6","MPI-ESM1-2-LR","CanESM5"]
folder="/gpfs/scratch/bsc32/bsc400019/"
outfolder=folder+"attribution/skill/"
os.makedirs(outfolder,exist_ok=True)
#same relative uncertainty levels as the cumulative area curves in Fig. 3a
if varn=="tas":
	threshs=[(x+1)*10 for x in range(10)]
else:
	threshs=[(x+1)*10 for x in range(20)]
tag=varn+"_"+metric+"_block"+str(W)+"_"+str(relerr_cutoff)+endyr

#load large ensemble and bootstrapped coefficients, file conventions as in plot_04_Fig3_uncertainty_ag.py
def load_coefs(model):
	if "0.99" in metric:
		coefs=[]
		for i in range(50):
			coefs.append(xr.open_dataset(folder + "attribution/attr_coefs_" + model + "_" + varn + "_" + metric + "_" + str(i) + ".nc"))
		coefs=xr.concat(coefs,dim="member").coef
		files=np.sort(glob.glob(folder + "attribution/bootstrap/attr_coefs_" + model + "_" + varn + "_" + metric + "_block"+str(W)+"_seed*.nc"))
		coefs_bs=xr.concat([xr.open_dataset(f).coef for f in files],dim="sample")
	else:
		coefs=xr.open_dataset(folder + "attribution/attr_coefs_" + model + "_" + varn + "_" + metric + endyr + ".nc").coef
		coefs_bs=xr.open_dataset(folder + "attribution/bootstrap/attr_coefs_" + model + "_" + varn + "_" + metric + "_block"+str(W) + endyr + ".nc").coef
	coefs=coefs.sortby("lon").sortby("lat").transpose("member","lat","lon").load()
	coefs_bs=coefs_bs.sortby("lon").sortby("lat").transpose("sample","member","lat","lon").load()
	assert np.allclose(coefs.lat,coefs_bs.lat) and np.allclose(coefs.lon,coefs_bs.lon), "grids differ"
	coefs_bs=coefs_bs.assign_coords(lat=coefs.lat,lon=coefs.lon)
	#order the LE members as the bootstrapped members, so member n of both is the same realisation
	coefs=coefs.sel(member=coefs_bs.member.values)
	print(model + ": " + str(coefs.member.size) + " LE members, " + str(coefs_bs.sample.size) + " bootstrap samples x " + str(coefs_bs.member.size) + " members")
	return(coefs,coefs_bs)

def relerr(std,av):
	return(100*std/np.abs(av))

def cap(da):
	#capping relerr because it diverges where coefficients ~0 (non-finite values set to the cap, as in plot_04)
	return(da.where(da<relerr_cutoff,relerr_cutoff))

def area_weights(da):
	#cos(lat) weights flattened in (lat,lon) order, matching flat()
	return(np.repeat(np.cos(np.deg2rad(da.lat.values)),da.lon.size))

#weighted pattern correlation between rows of A (k x N) and rows of B (l x N) -> (k x l)
def corr_matrix(A,B,w):
	w=w/w.sum()
	A=A-(A*w).sum(axis=1,keepdims=True)
	B=B-(B*w).sum(axis=1,keepdims=True)
	A=A/np.sqrt((A**2*w).sum(axis=1,keepdims=True))
	B=B/np.sqrt((B**2*w).sum(axis=1,keepdims=True))
	return((A*w)@B.T)

def wrmse(A,B,w):
	return(np.sqrt(((A-B)**2*w).sum(axis=-1)/w.sum()))

def flat(da):
	#(member,lat,lon) -> (member, lat*lon)
	return(da.values.reshape(da.shape[0],-1))

#grid cell bounds for conservative regridding (Gaussian/regular lat-lon grids, poles closed)
def with_bounds(da):
	lat=da.lat.values
	lon=da.lon.values
	lat_b=np.concatenate(([-90],(lat[1:]+lat[:-1])/2,[90]))
	dlon=np.mean(np.diff(lon))
	lon_b=np.concatenate(([lon[0]-dlon/2],(lon[1:]+lon[:-1])/2,[lon[-1]+dlon/2]))
	return(xr.Dataset(coords={"lat":lat,"lon":lon,"lat_b":lat_b,"lon_b":lon_b}))

LE={}
BS={}
for m, model in enumerate(models):
	coefs,coefs_bs=load_coefs(model)
	LE_av=coefs.mean(dim="member")
	LE_std=coefs.std(dim="member")
	BS_av=coefs_bs.mean(dim="sample")
	BS_std=coefs_bs.std(dim="sample")
	LE[model]={"coefs":coefs,"av":LE_av,"std":LE_std,"chi":relerr(LE_std,LE_av)}
	BS[model]={"av":BS_av,"std":BS_std,"chi":relerr(BS_std,BS_av)}
	del coefs_bs

rows=[]
maps=[]
for m, model in enumerate(models):
	coefs=LE[model]["coefs"]
	chi_LE=LE[model]["chi"]
	chi_BS=BS[model]["chi"]
	w=area_weights(chi_LE)
	nmem=chi_BS.member.size

	#observed skill on native grid (as Fig. 3b), against the full LE and a leave-one-out LE excluding the bootstrapped member
	A=flat(cap(chi_BS))
	r_obs=corr_matrix(A,cap(chi_LE).values.reshape(1,-1),w)[:,0]
	chi_LOO=[]
	for n in range(nmem):
		c=coefs.drop_isel(member=n)
		chi_LOO.append(relerr(c.std(dim="member"),c.mean(dim="member")))
	chi_LOO=xr.concat(chi_LOO,dim="member")
	r_obs_LOO=np.array([corr_matrix(A[n:n+1],flat(cap(chi_LOO))[n:n+1],w)[0,0] for n in range(nmem)])
	rmse_obs=wrmse(A,cap(chi_LE).values.reshape(1,-1),w)

	#(2) noise ceiling
	#reliability of the LE map: correlation of chi from random 25/25 member splits, stepped up to 50 with Spearman-Brown
	rng=np.random.default_rng(0)
	r_half=[]
	for s in range(nsplit):
		perm=rng.permutation(coefs.member.size)
		h1=coefs.isel(member=perm[:coefs.member.size//2])
		h2=coefs.isel(member=perm[coefs.member.size//2:])
		c1=cap(relerr(h1.std(dim="member"),h1.mean(dim="member"))).values.reshape(1,-1)
		c2=cap(relerr(h2.std(dim="member"),h2.mean(dim="member"))).values.reshape(1,-1)
		r_half.append(corr_matrix(c1,c2,w)[0,0])
	r_half=np.array(r_half)
	r_LE=2*np.mean(r_half)/(1+np.mean(r_half))
	#reliability of single-realisation bootstrap maps: correlation between bootstrap maps of different members
	R=corr_matrix(A,A,w)
	r_boot=R[np.triu_indices(nmem,k=1)]
	ceiling=np.sqrt(np.median(r_boot)*r_LE)

	for n in range(nmem):
		rows.append({"model":model,"member":str(chi_BS.member.values[n]),"r_obs":r_obs[n],"r_obs_LOO":r_obs_LOO[n],"rmse_obs":rmse_obs[n]})
	nc_stats={"r_half_median":np.median(r_half),"r_LE50":r_LE,"r_boot_median":np.median(r_boot),"r_boot_p5":np.percentile(r_boot,5),"r_boot_p95":np.percentile(r_boot,95),"ceiling":ceiling,"r_obs_median":np.median(r_obs),"r_obs_LOO_median":np.median(r_obs_LOO)}
	pd.DataFrame([nc_stats],index=[model]).to_csv(outfolder+"noise_ceiling_"+model+"_"+tag+".csv")
	np.save(outfolder+"r_boot_pairs_"+model+"_"+tag+".npy",r_boot)
	np.save(outfolder+"r_half_"+model+"_"+tag+".npy",r_half)
	print(model + " noise ceiling: " + str({k:round(float(v),3) for k,v in nc_stats.items()}))

	#(3) threshold skill: cells the bootstrap classes as chi<X vs. cells truly chi<X in the full LE (area weighted), as in Fig. 3
	for t, thresh in enumerate(threshs):
		truth=np.repeat(chi_LE.values.reshape(1,-1),nmem,axis=0)<thresh
		pred=flat(chi_BS)<thresh
		tp=((truth&pred)*w).sum(axis=1)
		for n in range(nmem):
			rows_t={"model":model,"member":str(chi_BS.member.values[n]),"thresh":thresh,
				"precision":tp[n]/(pred[n]*w).sum() if pred[n].any() else np.nan,
				"hit_rate":tp[n]/(truth[n]*w).sum() if truth[n].any() else np.nan,
				"base_rate":(truth[n]*w).sum()/w.sum(),
				"pred_area":(pred[n]*w).sum()/w.sum()}
			maps.append(rows_t)

	#(4) bias maps vs. full LE, as in Fig. 3: log ratios of chi, sigma and |mu| (log chi ratio = log sigma ratio - log |mu| ratio)
	#signed mu ratio as check that the |mu| ratio reflects noise in single-realisation estimates rather than a bias in the forced response
	ratio_chi=chi_BS/chi_LE
	bias=xr.Dataset()
	bias["logratio_chi"]=np.log(ratio_chi).where(np.isfinite(np.log(ratio_chi))).median(dim="member")
	bias["logratio_std"]=np.log(BS[model]["std"]/LE[model]["std"]).median(dim="member")
	bias["logratio_absmu"]=np.log(np.abs(BS[model]["av"])/np.abs(LE[model]["av"])).where(lambda x:np.isfinite(x)).median(dim="member")
	bias["ratio_mu_signed"]=(BS[model]["av"]/LE[model]["av"]).median(dim="member")
	bias["mean_mu_BS_over_LE"]=BS[model]["av"].mean(dim="member")/LE[model]["av"]
	bias["frac_within25"]=(np.abs(ratio_chi-1)<0.25).mean(dim="member")
	bias["chi_LE"]=chi_LE
	bias["chi_BS_median"]=chi_BS.median(dim="member")
	bias.to_netcdf(outfolder+"bias_maps_"+model+"_"+tag+".nc")
	#per-member bootstrap chi maps (uncapped) for the cumulative area curves of Fig. 3a
	chi_BS.rename("chi").to_netcdf(outfolder+"chi_bootstrap_"+model+"_"+tag+".nc")

pd.DataFrame(rows).to_csv(outfolder+"obs_skill_"+tag+".csv",index=False)
pd.DataFrame(maps).to_csv(outfolder+"threshold_skill_"+tag+".csv",index=False)

#(1) cross-model benchmark: each bootstrapped model is evaluated on its own native grid, with the capped chi maps of the
#other large ensembles conservatively regridded onto it, so the diagonal equals the native-grid correlations of Fig. 3
cross=[]
for i, model_bs in enumerate(models):
	target=with_bounds(LE[model_bs]["chi"])
	LEmaps=[]
	for model_le in models:
		if model_le==model_bs:
			LEmaps.append(cap(LE[model_le]["chi"]).values.flatten())
		else:
			regridder=xe.Regridder(with_bounds(LE[model_le]["chi"]),target,"conservative",periodic=True)
			LEmaps.append(regridder(cap(LE[model_le]["chi"])).values.flatten())
	LEmaps=np.array(LEmaps)
	w=area_weights(LE[model_bs]["chi"])
	A=flat(cap(BS[model_bs]["chi"]))
	R=corr_matrix(A,LEmaps,w)
	E=np.array([wrmse(A,LEmaps[j:j+1],w) for j in range(len(models))]).T
	for n in range(A.shape[0]):
		for j, model_le in enumerate(models):
			cross.append({"model_bs":model_bs,"member":n,"model_le":model_le,"r":R[n,j],"rmse":E[n,j],
				"best_r":models[np.argmax(R[n])]==model_le,"best_rmse":models[np.argmin(E[n])]==model_le})
cross=pd.DataFrame(cross)
cross.to_csv(outfolder+"cross_model_"+tag+".csv",index=False)

#(1b) robustness test: all capped chi maps (large ensembles and bootstrapped members, incl. the own model) conservatively
#regridded to a common grid (the coarsest, CanESM5), so that every comparison is between maps processed the same way
target=with_bounds(LE["CanESM5"]["chi"])
common={}
for model in models:
	if model=="CanESM5":
		common[model]={"LE":cap(LE[model]["chi"]),"BS":cap(BS[model]["chi"])}
	else:
		regridder=xe.Regridder(with_bounds(LE[model]["chi"]),target,"conservative",periodic=True)
		common[model]={"LE":regridder(cap(LE[model]["chi"])),"BS":regridder(cap(BS[model]["chi"]))}
w=area_weights(common["CanESM5"]["LE"])
LEmaps=np.array([common[model]["LE"].values.flatten() for model in models])
cross_common=[]
for i, model_bs in enumerate(models):
	A=flat(common[model_bs]["BS"])
	R=corr_matrix(A,LEmaps,w)
	E=np.array([wrmse(A,LEmaps[j:j+1],w) for j in range(len(models))]).T
	for n in range(A.shape[0]):
		for j, model_le in enumerate(models):
			cross_common.append({"model_bs":model_bs,"member":n,"model_le":model_le,"r":R[n,j],"rmse":E[n,j],
				"best_r":models[np.argmax(R[n])]==model_le,"best_rmse":models[np.argmin(E[n])]==model_le})
cross_common=pd.DataFrame(cross_common)
cross_common.to_csv(outfolder+"cross_model_common_grid_"+tag+".csv",index=False)
for lab, c in [("native grid of bootstrapped model",cross),("common CanESM5 grid",cross_common)]:
	own=c[c.model_bs==c.model_le]
	print("cross-model comparison on " + lab + ": " + str(int(own.best_r.sum())) + "/" + str(len(own)) + " closest to own model (pattern correlation), " + str(int(own.best_rmse.sum())) + "/" + str(len(own)) + " (RMSE)")
	print(c.groupby(["model_bs","model_le"]).r.median().unstack().loc[models,models].round(3))

print(cross.groupby(["model_bs","model_le"])[["r","rmse"]].median().unstack())
own=cross[cross.model_bs==cross.model_le]
print("correctly identified by pattern correlation: " + str(int(own.best_r.sum())) + "/" + str(len(own)))
print("correctly identified by RMSE: " + str(int(own.best_rmse.sum())) + "/" + str(len(own)))
print(pd.DataFrame(maps).groupby(["model","thresh"])[["precision","hit_rate","base_rate"]].median())
