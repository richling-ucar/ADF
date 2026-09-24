"""
adf_histogram

Create histogram(s) of all specified (2D) variables.

- Constructs global histograms: total, ocean, land.
  + for consistency with other ADF, will default to making annual and seasonal histograms.
- Saves the resulting histogram data as netCDF (into the plots directory by default)
- Plots the histograms.

Options:
  - specify whether to do spatial distribution from climo files
    or combine temporal and spatial using time series files

Possible future enhancements:
  - specify a landmask file
  - specific histogram options in the variable defaults file (looks for hist_bins)
  - ability to specify additional regions (?)
  - specify the output location for netCDF files
  - improved detection of dimensions that (like cosp height)
  - allow for rgridding; right now if LANDFRAC and data are not the same, just skips.

"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import sys, time, json
from glob import glob

import cam_era5_functions as cfun
import cam_diagnostic as cdog

import warnings  # use to warn user about missing files.
import adf_utils as utils
warnings.formatwarning = utils.my_formatwarning

#
# USER-ADJUSTABLE PARAMETERS:
#

def load_nc_cldrgme(file_paths):
    """
    Load and combine NetCDF cloud regime files into a single DataFrame.
    
    Parameters
    ----------
    file_paths : list
        List of file paths to NetCDF files
        
    Returns
    -------
    pd.DataFrame
        Combined dataframe with all blocks
    """
    combined_blocks = []   
    for path in file_paths:
        ds = xr.open_dataset(path)
        
        # Convert to DataFrame for convenient filtering
        df = ds.to_dataframe().reset_index().drop(columns="index")
        
        # Get all unique (label, index) pairs
        unique_blocks = df[["block_label", "block_index"]].drop_duplicates()
        
        # Loop through each unique block
        for _, row in unique_blocks.iterrows():
            label = row["block_label"]
            idx = row["block_index"]
        
            # Filter DataFrame
            df_block = df[(df["block_label"] == label) & (df["block_index"] == idx)].copy()
        
            combined_blocks.append(df_block)
    
    all_blocks = pd.concat(combined_blocks, ignore_index=True)
    return all_blocks

def inform_model_analysis(adfobj):
    """
    Run the INFORM campaign cloud micro analysis.
    
    Parameters
    ----------
    adfobj : ADF object
        The ADF diagnostics object containing campaign information.
    """
    print('\nStarting INFORM campaign cloud micro analysis.')
    
    # Load cloud regime data for the specified campaign
    campaign_name = "SOCRATES"
    print(adfobj.plotting_scripts)
    #print(sorted(glob(f"{__file__}/*.py")))
    
    micro_diag_dir = Path("scripts/plotting/model_analysis_micro_scripts/")
    micro_all_files = sorted(micro_diag_dir.glob(f"*.py"))
    print(micro_all_files)
    micro_filtered_files = [f for f in micro_all_files if f.name != '__init__.py']
    print("\nMICRO WOWSA:  -> ",micro_filtered_files,"\n-------------------------------------\n")

    macro_diag_dir = Path("scripts/plotting/model_analysis_macro_scripts/")
    macro_all_files = sorted(macro_diag_dir.glob(f"*.py"))
    print(macro_all_files)
    macro_filtered_files = [f for f in macro_all_files if (f.name != '__init__.py') or (f.name != 'cam_obs_dropsonde_comp_ryan.py')]
    print("\nMACRO WOWSA:  -> ",macro_filtered_files,"\n-------------------------------------\n")

    process_diag_dir = Path("scripts/plotting/model_analysis_process_scripts/")
    process_all_files = sorted(process_diag_dir.glob(f"*.py"))
    print(process_all_files)
    process_filtered_files = [f for f in process_all_files if f.name != '__init__.py']
    print("\PROCESS WOWSA:  -> ",process_filtered_files,"\n-------------------------------------\n")
    

    plot_dict = {"micro":"",
                 "macro":"",
                 "process":""}

    for i,item in enumerate(adfobj.plotting_scripts):
        if type(item)==dict:
            #print(item,item.keys())
            if list(item.keys())[0] == "inform_model_analysis":
                req_diags = []
                print('item["inform_model_analysis"] WHAT?',item["inform_model_analysis"])
                for diag_type in item["inform_model_analysis"]:
                    if diag_type in ["micro", "macro", "process"]:
                        #lst.index("dict")
                        req_diags.append(diag_type)

    print(req_diags)
    #sys.exit()
    # Additional analyses can be added here as needed
    print('\nStarting COMPASS model analysis plots.')
    #print("adfobj.campaigns_dict",adfobj.campaigns_dict)
    campaigns = adfobj.campaigns_dict["name"]
    #cam_paths = models_dict.keys()
    campaigns_defaults = adfobj.campaigns_dict
    #print("campaigns_defaults",campaigns_defaults)
    #models_dict = adfobj.model_dict
    cam_hist_locs = adfobj.get_cam_info("cam_hist_loc")
    #for cam_path in ["/glade/derecho/scratch/islas/archive/f.e30_cam6_4_120.FHIST_BGC.f09_f09_mg17.SOCRATES_nudgeUVTfull_withCOSP_tau6h.002/atm/hist"]:
    for idx,campaign in enumerate(campaigns):
        #composited_data_path = Path(campaigns_defaults[campaign]["composited_data"])
        composited_data_path = Path(campaigns_defaults["composited_data"][idx])
        # Find all flight data files for this campaign
        composited_data_path.mkdir(parents=True, exist_ok=True)
        file_paths = sorted(glob(f"{composited_data_path}/*{campaign}*RF*.nc"))
        if not file_paths:
            print(f"No files found for campaign {campaign} in path {composited_data_path}???")
            sys.exit(0)

        if campaign == 'SOCRATES':
            lat_slice = slice(-62, -42)
            lon_slice = slice(135, 165)   # 0..360 convention
        elif campaign == 'CSET':
            lat_slice = slice(15, 45)
            lon_slice = slice(200, 240)

        # Load combined flight data
        #All_rf_df = pd.read_csv(f"{composited_data_path}/{campaign}_combined_RF_data.csv")
        if not Path(f"{composited_data_path}/{campaign}_combined_RF_data.csv").is_file():
            All_rf_df = load_nc_cldrgme(file_paths)
            All_rf_df.to_csv(f"{composited_data_path}/{campaign}_combined_RF_data.csv", index=False)
        else:
            All_rf_df = pd.read_csv(f"{composited_data_path}/{campaign}_combined_RF_data.csv")

        df_sonde = None
        if "macro" in req_diags:
            sonde_path = campaigns_defaults["composited_data"][idx]
            ds_sonde = xr.open_dataset(f'{sonde_path}/{campaign}_sonde_data_composite.nc')
            df_sonde = ds_sonde.to_dataframe()

        air_time = All_rf_df.Time
        time_ser = air_time.astype('datetime64[ns]')
        #print("time_ser",time_ser,"\n")
        
        tmin = pd.to_datetime(time_ser.min()).to_pydatetime()
        #print("time_ser.index(time_ser.min())",list(time_ser).index(time_ser.min()))
        
        ###tmax = time_ser[list(time_ser).index(time_ser.min())+1200]
        #tmax = pd.to_datetime(time_ser.mean()).to_pydatetime()
        tmax = pd.to_datetime(time_ser.max()).to_pydatetime()

        #tmin = time_ser[list(time_ser).index(time_ser.min())+12000]
        #tmax = time_ser[list(time_ser).index(time_ser.min())+12000+24000]

        print("tmin",tmin)
        print("tmax",tmax,"\n")
        t0 = time.perf_counter()
        ds_era5 = cfun.load_era5_files(tmin, tmax, lat_slice, lon_slice)
        t0 = utils.timer("\nloaded ds_era5", t0)

        ds_cams = {}
        ds_outs = {}
        for idx,cam_path in enumerate(cam_hist_locs):
            cam_dir = Path(cam_path)
            print("\ncam_dir",cam_dir)
            cam_desc = adfobj.get_cam_info("case_nickname")[idx]
            print("cam_desc",cam_desc)
            print('adfobj.hist_string["test_hist_str"]',adfobj.hist_string["test_hist_str"])
            print('adfobj.hist_string["test_hist_str"][idx]',adfobj.hist_string["test_hist_str"][idx])
            file_sel = adfobj.hist_string["test_hist_str"][idx][0]

            print("CAM case description:",cam_desc)
            print("CAM case history num:",file_sel,"\n")

            #"""
            ds_cam = cfun.load_cam_files(cam_dir, file_sel, tmin, tmax, campaign=campaign)
            t0 = utils.timer("loaded ds_cam", t0)
            """if not Path(f"{composited_data_path}/{campaign}_CAM_microphys_composite.nc").is_file():
                # Create datasets for CAM state, microphysics and aerosol concentrations based on aircraft instrumentation
                ds_out = cfun.compute_cam_aerosol_micphys_metrics(ds_cam, lat_slice=lat_slice, lon_slice=lon_slice) # dataset with all the microphysics, state, and aerosol parameters
                t0 = utils.timer("  loaded microphys_composite: ds_out", t0)
                ds_out.to_netcdf(f"{composited_data_path}/{campaign}_CAM_microphys_composite.nc")
            else:
                print('loaded microphys_composite FROM FILE')
                ds_out = xr.open_dataset(f"{composited_data_path}/{campaign}_CAM_microphys_composite.nc")"""
            
            ds_out = cfun.compute_cam_aerosol_micphys_metrics(ds_cam, lat_slice=lat_slice, lon_slice=lon_slice) # dataset with all the microphysics, state, and aerosol parameters
            t0 = utils.timer("loaded ds_out", t0)
            # Create dataset of CAM cloud controlling factors for compositing of CAM data similar to sonde/aircraft observations
            ds_cam_ccfs = cfun.load_cam_ccfs(ds_cam, lat_slice=lat_slice, lon_slice=lon_slice) # dataset used for compositing above dataset
            t0 = utils.timer("loaded ds_cam_ccfs", t0)

            """
            fils = [f"{composited_data_path}/{campaign}_CAM_microphys_strat_composite.nc",
                    f"{composited_data_path}/{campaign}_CAM_microphys_open_composite.nc",
                    f"{composited_data_path}/{campaign}_CAM_microphys_mask_strat_composite.nc",
                    f"{composited_data_path}/{campaign}_CAM_microphys_mask_open_composite.nc",]

            good = []
            for fil in fils:
                if Path(fil).is_file():
                    good.append(fil)
            if len(good) == len(fils):
            #if Path(f"{composited_data_path}/{campaign}_CAM_microphys_strat_composite.nc").is_file():
                ds_strat = xr.open_dataset(f"{composited_data_path}/{campaign}_CAM_microphys_strat_composite.nc")
                ds_open = xr.open_dataset(f"{composited_data_path}/{campaign}_CAM_microphys_open_composite.nc")
                mask_strat = xr.open_dataset(f"{composited_data_path}/{campaign}_CAM_microphys_mask_strat_composite.nc")
                mask_open = xr.open_dataset(f"{composited_data_path}/{campaign}_CAM_microphys_mask_open_composite.nc")
                ds_cam_comp = {
                    "strat": ds_strat,   # full ds with Qc, Qr, … masked to strat condition
                    "open":  ds_open,    # full ds with Qc, Qr, … masked to open condition
                    "masks": {"strat": mask_strat, "open": mask_open},
                }
            else:
                ds_cam_comp = cfun.subset_cam_by_campaign(ds_out,ds_cam_ccfs,campaign=campaign, nc_savepath=composited_data_path)
            """
            ds_cam_comp = cfun.subset_cam_by_campaign(ds_out,ds_cam_ccfs,campaign=campaign, nc_savepath=composited_data_path)
            t0 = utils.timer("loaded ds_cam_comp", t0)

            """
            ds_cam = cfun.load_cam_files(cam_dir, file_sel, tmin, tmax, campaign=campaign)
            t0 = utils.timer("calc ds_cam", t0)
            ds_out = cfun.compute_cam_aerosol_micphys_metrics(ds_cam, lat_slice=lat_slice, lon_slice=lon_slice) # dataset with all the microphysics, state, and aerosol parameters
            t0 = utils.timer("calc ds_out", t0)

            # Create dataset of CAM cloud controlling factors for compositing of CAM data similar to sonde/aircraft observations
            ds_cam_ccfs = cfun.load_cam_ccfs(ds_cam, lat_slice=lat_slice, lon_slice=lon_slice) # dataset used for compositing above dataset
            t0 = utils.timer("calc ds_cam_ccfs", t0)
            # Composite the CAM data using thresholds defined by aircraft observations.
            ## Use Cluster running this for CAM7
            ds_cam_comp = cfun.subset_cam_by_campaign(ds_out,ds_cam_ccfs,campaign=campaign)
            t0 = utils.timer("calc ds_cam_comp", t0)

            """

            ds_cams[cam_desc] = ds_cam_comp
            ds_outs[cam_desc] = ds_out
            
            #cam_obs_nd_lwc_pdf(adfobj, All_rf_df, ds_cams[cam_desc], campaign)
            #sensitivity_vs_sigmaw_2x2_obs_cam(adfobj, All_rf_df, ds_cam_comp, campaign, cam_desc)
            #t0 = utils.timer("  plotted alpha_sigmaw_regime_comparison", t0)
            #cam_obs_dropsonde_comp(adfobj, campaign, df_sonde, ds_outs[cam_desc], ds_era5)
        print("\n\n",ds_outs.keys(),"\n\n")
        #if campaign == "SOCRATES":
            #sensitivity_vs_sigmaw_2x2_obs_cam(adfobj, All_rf_df, ds_cams, campaign, cam_desc)
            #alpha_sigmaw_regime_comparison_2x2(adfobj, All_rf_df, ds_cams, campaign, cam_desc)
        ###cam_obs_dropsonde_comp(adfobj, campaign, df_sonde, ds_outs, ds_era5)
        #cam_obs_nd_lwc_pdf(adfobj, All_rf_df, ds_cams, campaign, cam_desc)
        
        for diag in req_diags:
            if diag == "macro":
                from model_analysis_macro_scripts import cam_obs_dropsonde_comp
                cam_obs_dropsonde_comp(adfobj, campaign, df_sonde, ds_outs, ds_era5)
            if diag == "micro":
                from model_analysis_micro_scripts import cam_obs_nd_lwc_pdf
                cam_obs_nd_lwc_pdf(adfobj, All_rf_df, ds_cams, campaign)
            if campaign == "SOCRATES":
                if diag == "process":
                    from model_analysis_process_scripts import alpha_sigmaw_regime_comparison_2x2,sensitivity_vs_sigmaw_2x2_obs_cam
                    sensitivity_vs_sigmaw_2x2_obs_cam(adfobj, All_rf_df, ds_cams, campaign)
                    alpha_sigmaw_regime_comparison_2x2(adfobj, All_rf_df, ds_cams, campaign, cam_desc)

        #for a in req_diags:
        #    for b in plot_dict[a]:
    print("ok yeah")