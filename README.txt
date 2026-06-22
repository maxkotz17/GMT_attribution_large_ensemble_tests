This repository contains code used to run the analysis for the paper:

"Attribution of climate extremes with global temperature correlations: accounting for uncertainty from internal variability" 
Maximilian Kotz & Markus Donat

Processed climate data is not included because of size constraints on github (climate data from climate model runs>20GB, and bootstrapped data correlation coefficients>20GB), but is stored on the supercomputing infrastucture of the Barcelona Supercomputing Centre and can be accessed from the authors if requested.

Scripts:
01_calc_ensemble_GMT.py: calculation of global mean temperature from large ensemble climate model runs.

02_calc_climperciles.py: calculation of extreme percentiles from large ensemble climate model runs.

03_calc_extexp.py: calculation of exposure to climate extreme percentiles from large ensemble climate model runs.

04_calc_TPX.py: calculation of hottest 5 day period of a year for temperature and precipitation from large ensemble climate model runs.

05_attr_autocorr_poiss_ens_membs.py: attribution of climate extremes using Poisson correlation with global temperatures (without bootstrapping) and calculation of residuals of this correlation within the large ensemble climate model runs.

06_attr_autocorr_lin_ens_membs.py: attribution of climate extremes using linear correlation with global temperatures (without bootstrapping) and calculation of residuals of this correlation within the large ensemble climate model runs.

07_bootstrap_ensemble_members.py: bootstrapped attribution using Poisson correlation with global temperatures (without bootstrapping) and calculation of residuals of this correlation within the large ensemble climate model runs.

08_bootstrap_ensemble_members_lin.py: bootstrapped attribution using linear correlation with global temperatures (without bootstrapping) and calculation of residuals of this correlation within the large ensemble climate model runs.

09_calc_ERA5_GMT.py: calculation of global mean temperature from ERA5 reanalysis.

10_calc_ERA5_perciles.py: calculation of exposure to climate extreme percentiles from ERA5 reanalysis.

11_calc_ERA5_extexp.py: calculation of exposure to climate extreme percentiles from ERA5 reanalysis.

12_calc_ERA5_TPX.py: calculation of hottest 5 day period of a year for temperature and precipitation from ERA5 reanalysis.

13_attr_autocorr_poiss_ERA5.py: attribution of climate extremes using Poisson correlation with global temperatures (without bootstrapping) and calculation of residuals of this correlation within the ERA5 reanalysis.

14_attr_autocorr_lin_ERA5.py: attribution of climate extremes using linearcorrelation with global temperatures (without bootstrapping) and calculation of residuals of this correlation within the ERA5 reanalysis.

15_bootstrap_ERA5.py: bootstrapped attribution using Poisson correlation with global temperatures (without bootstrapping) and calculation of residuals of this correlation within the ERA5 reanalysis.

16_bootstrap_ERA5_lin.py: bootstrapped attribution using linear correlation with global temperatures (without bootstrapping) and calculation of residuals of this correlation within the ERA5 reanalysis.

Plotting scripts:

plot_01_Fig1.py

plot_02_Fig2_bootstrap_uncertainty_summary.py

plot_03_autocorr_model.py

plot_04_Fig3_uncertainty_ag.py

plot_05_Fig6_ERA5_attr_example.py
 
