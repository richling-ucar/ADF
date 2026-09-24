"""
Flight track cloud regime analysis for COMPASS campaigns.
Generates map with flight tracks and cloud regime pie chart.
"""

import os, xarray as xr
from pathlib import Path

#from dask_jobqueue import PBSCluster
#from dask.distributed import Client
#import dask
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
xr.set_options(file_cache_maxsize=1)       # do NOT set 0 on your xarray version
# from dask import compute as dask_compute
#import dask.array as da

import cam_diagnostic as cdog

#cloud regimes
def alpha_sigmaw_regime_comparison_2x2(adf, All_rf_df, ds_cams, campaign, cam_desc):
    # Example usage of the microphysics processes plotting function
    # Adjust the variable names and datasets as needed

    #Notify user that script has started:
    msg = "\n  Generating CAM-Obs Sigma W Regime Comparison plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")
    fig, axes, out = cdog.plot_alpha_sigmaw_regime_comparison_2x2(
            df_obs=All_rf_df,
            cam_cases=ds_cams,
            Nd_col="CONCD_RWIO",
            lwc_col="PLWCD_RWIO",
            CCN_col="Nccn_UH_CVIU",
            sigmaw_col="Sigma_w",
            regime_col="cloud_regime",
            Ndriz_col="Ndriz_2DC",
            drizzle_threshold=0.0,
            cam_temp_threshold_K=268.15,
            cam_Nd_var="cam_Nc",
            cam_CCN_var="cam_N_UHSAS",
            cam_sigmaw_var="sigma_w",
            cam_rain_var="cam_Nr",
            cam_rain_threshold=0.0,
            cam_regime_map={"Stratocumulus":"strat", "Open-Cell":"open"},
            n_bins=10,
            min_n=50,
            binning="quantile",
            figsize=(10, 8),
        )
    print("len(ds_cams.keys())",len(ds_cams.keys()))
    if len(ds_cams.keys()) > 1:
        img_base = f'{campaign}__sigmaw_regime_comparison__multi_CAM'
    else:
        img_base = f'{campaign}__sigmaw_regime_comparison__{cam_desc}'
    img_name = f'{img_base}.png'
    plot_loc = Path(adf.plot_location)
    img_path = f"{plot_loc}/{img_name}"
    fig.savefig(img_path, dpi=300, bbox_inches='tight')
    adf.add_website_data(str(img_path), img_base, campaign, plot_type="Sigma W Regime Comparison",
                            category="Model Micro State", plot_loc=str(plot_loc),cam_desc=cam_desc)