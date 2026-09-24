"""
Flight track cloud regime analysis for COMPASS campaigns.
Generates map with flight tracks and cloud regime pie chart.
"""

from pathlib import Path
import time
import json
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

#from dask_jobqueue import PBSCluster
#from dask.distributed import Client
#import dask
#os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
#xr.set_options(file_cache_maxsize=1)       # do NOT set 0 on your xarray version
# from dask import compute as dask_compute
#import dask.array as da

import cam_diagnostic as cdog
import adf_utils as utils

def disk_size(obj, name="data"):
    """
    Report an object's size in MB for debug logging, computed in-memory
    (no disk I/O). For a dask-backed xr.DataArray/Dataset, ".nbytes" is
    shape/dtype metadata only, so this does NOT force a compute.
    """
    if isinstance(obj, np.ndarray):
        size = obj.nbytes

    elif isinstance(obj, pd.DataFrame):
        size = obj.memory_usage(deep=True).sum()

    elif isinstance(obj, (xr.DataArray, xr.Dataset)):
        size = obj.nbytes

    else:
        raise TypeError(f"Unsupported type: {type(obj)}")

    print(f"{name}: {size / 1e6:.2f} MB")
    return size


def _profile_records(p, mean, std):
    """A regime's profile as tidy per-level records: [{p_hPa, mean, std}, ...]."""
    return [
        {"p_hPa": float(p_), "mean": float(m_), "std": float(s_)}
        for p_, m_, s_ in zip(p, mean, std)
    ]


def _bias_records(p, delta):
    """A regime's bias-vs-Obs as tidy per-level records: [{p_hPa, delta}, ...]."""
    return [
        {"p_hPa": float(p_), "delta": float(d_)}
        for p_, d_ in zip(p, delta)
    ]


def _named_profile_entry(p_open, mean_open, std_open, p_strat, mean_strat, std_strat,
                          N_open=None, N_strat=None,
                          delta_open=None, delta_strat=None):
    """
    One named source's open/strat profiles - a CAM case, or a shared
    reference like "Obs"/"ERA5". N_open/N_strat (drop counts) are an
    observation property, so they're only passed for the "Obs" entry.
    delta_open/delta_strat (bias vs Obs, already computed in Python - see
    compute_case_stats) are only passed for actual CAM cases, since Obs/
    ERA5 don't have a bias against themselves.
    """
    entry = {
        "open":  _profile_records(p_open, mean_open, std_open),
        "strat": _profile_records(p_strat, mean_strat, std_strat),
    }
    if N_open is not None:
        entry["N_open"] = N_open
        entry["N_strat"] = N_strat
    if delta_open is not None:
        entry["open_bias"] = _bias_records(p_open, delta_open)
        entry["strat_bias"] = _bias_records(p_strat, delta_strat)
    return entry


def save_named_profile_json(name, campaign, out_dir, p_open, mean_open, std_open,
                             p_strat, mean_strat, std_strat,
                             N_open=None, N_strat=None,
                             delta_open=None, delta_strat=None):
    """
    Write one named profile (a CAM case, or a shared reference like "Obs"/
    "ERA5") to a browser-friendly JSON file, for use outside of Python (e.g. D3).
    Filenames are qualified by campaign since "Obs"/"ERA5" (and possibly
    case names) are reused across campaigns - without this, a second
    campaign's run would silently overwrite the first's per-source files.
    """
    record = {"case": name, **_named_profile_entry(
        p_open, mean_open, std_open, p_strat, mean_strat, std_strat,
        N_open, N_strat, delta_open, delta_strat,
    )}
    out_path = Path(out_dir) / f"sonde_profile_{campaign}_{name}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote profile JSON: {out_path}")


def save_dropsonde_dashboard_json(campaign, sonde_var, enough, out_dir, config_name):
    """
    Combine every source into the single-file shape the D3 dashboard expects
    (see lib/website_templates/examples/sample_dropsonde_profile.json):
    "Obs" and "ERA5" are peers of the CAM cases under "cases" - shared
    reference data, not duplicated inside every CAM case.
    """
    # Obs/ERA5 are shared across cases (same observations/reanalysis
    # regardless of which CAM run they're compared against), so grab them
    # from whichever case happens to be first - same convention the Python
    # plotting already uses (cam_ds_0 = enough[list(enough.keys())[0]]).
    first = next(iter(enough.values()))

    cases_json = {
        "Obs": _named_profile_entry(
            first["p_sonde_open"], first["sonde_open_mean"], first["sonde_open_std"],
            first["p_sonde_strat"], first["sonde_strat_mean"], first["sonde_strat_std"],
            first["N_open"], first["N_strat"],
        ),
        "ERA5": _named_profile_entry(
            first["p_era_open"], first["era_open_mean"], first["era_open_std"],
            first["p_era_strat"], first["era_strat_mean"], first["era_strat_std"],
        ),
    }
    for case, d in enough.items():
        cases_json[case] = _named_profile_entry(
            d["p_cam_open"], d["cam_open_mean"], d["cam_open_std"],
            d["p_cam_strat"], d["cam_strat_mean"], d["cam_strat_std"],
            delta_open=d["delta_open"], delta_strat=d["delta_strat"],
        )

    dashboard_json = {"campaign": campaign, "sonde_var": sonde_var, "cases": cases_json}

    out_path = Path(out_dir) / f"dropsonde_profiles_{campaign}_{config_name}.json"
    with open(out_path, "w") as f:
        json.dump(dashboard_json, f, indent=2)
    print(f"Wrote dropsonde dashboard JSON: {out_path}")


def compute_case_stats(case, campaign, out_dir, ds_out, ds_era5, df_sonde, sonde_var, cam_var, era_var):
    """
    Collocate model/obs profiles for a single case and return its "enough" entry.
    """
    # start from your open_mfdataset outputs
    ds_cam_sub = ds_out[[cam_var]].chunk({"time": -1, "lev": -1}).persist()
    ds_era = ds_era5[[era_var]].chunk({"time": -1, "level": -1}).persist()

    ### --- Open-Cell regime ---
    t0 = time.perf_counter()
    df_open, p_cam_open_raw, cam_arr_open, p_era_open_raw, era_arr_open = cdog.collocate_model_profiles_for_regime(
            df_sonde,
            ds_cam_sub, # CAM
            ds_era,     # ERA5
            regime_name="Open-Cell",
            sonde_var=sonde_var,
            cam_var=cam_var,
            era_var=era_var,
            cam_lat_name="lat",
            cam_lon_name="lon",
            cam_lev_name="lev",
            era_lat_name="latitude",
            era_lon_name="longitude",
            era_lev_name="level",
            to_percent_models=True,
        )
    t0 = utils.timer("loaded OPEN collocate_model_profiles_for_regime", t0)
    print("type(df_open",type(df_open))
    print("type(p_cam_open_raw",type(p_cam_open_raw))

    disk_size(p_cam_open_raw, 'p_cam_open_raw')

    disk_size(p_era_open_raw, 'p_era_open_raw')

    t0 = time.perf_counter()
    prof_sonde_open = cdog.sonde_profile(df_open, sonde_var, bin_hPa=10, to_percent=True)
    print("type(prof_sonde_open",type(prof_sonde_open))
    t0 = utils.timer("loaded OPEN sonde_profile", t0)
    disk_size(prof_sonde_open, 'prof_sonde_open')
    p_sonde_open    = prof_sonde_open["p_hPa"].values
    sonde_open_mean = prof_sonde_open["mean"].values
    sonde_open_std  = prof_sonde_open["std"].values

    p_cam_open, cam_open_mean, cam_open_std = cdog.mean_std_from_profiles(
            p_cam_open_raw, cam_arr_open
        )
    p_era_open, era_open_mean, era_open_std = cdog.mean_std_from_profiles(
            p_era_open_raw, era_arr_open
        )
    print("type(p_cam_open",type(p_cam_open))
    ### --- Stratocumulus regime ---
    df_strat, p_cam_strat_raw, cam_arr_strat, p_era_strat_raw, era_arr_strat = cdog.collocate_model_profiles_for_regime(
            df_sonde,
            ds_cam_sub,
            ds_era,
            regime_name="Stratocumulus",
            sonde_var=sonde_var,
            cam_var=cam_var,
            era_var=era_var,
            cam_lat_name="lat",
            cam_lon_name="lon",
            cam_lev_name="lev",
            era_lat_name="latitude",
            era_lon_name="longitude",
            era_lev_name="level",
            to_percent_models=True,
        )

    t0 = time.perf_counter()
    prof_sonde_strat = cdog.sonde_profile(df_strat, sonde_var, bin_hPa=10, to_percent=True)
    t0 = utils.timer("loaded STRATOCUMULUS sonde_profile", t0)
    p_sonde_strat    = prof_sonde_strat["p_hPa"].values
    sonde_strat_mean = prof_sonde_strat["mean"].values
    sonde_strat_std  = prof_sonde_strat["std"].values

    p_cam_strat, cam_strat_mean, cam_strat_std = cdog.mean_std_from_profiles(
            p_cam_strat_raw, cam_arr_strat
        )
    p_era_strat, era_strat_mean, era_strat_std = cdog.mean_std_from_profiles(
            p_era_strat_raw, era_arr_strat
        )

    # Separate dropsondes based on cloud regime
    # --- Counts of physical drops (RF, drop_num) in each regime ---
    N_open  = df_open.groupby(["RF", "drop_num"]).ngroups
    N_strat = df_strat.groupby(["RF", "drop_num"]).ngroups

    # Bias vs Obs (CAM - Obs), same calculation the Python plotting side
    # does (draw_open_bias/draw_strat_bias in lib/cam_diagnostic.py):
    # interpolate this case's own collocated Obs onto its own CAM pressure
    # grid, then subtract. Computed here (not in JS) so the D3 dashboard
    # just reads the result instead of recomputing it client-side.
    obs_open_interp = np.interp(
            p_cam_open[::-1], p_sonde_open[::-1], sonde_open_mean[::-1],
        )[::-1]
    delta_open = cam_open_mean - obs_open_interp

    obs_strat_interp = np.interp(
            p_cam_strat[::-1], p_sonde_strat[::-1], sonde_strat_mean[::-1],
        )[::-1]
    delta_strat = cam_strat_mean - obs_strat_interp

    dropsinde_dict = {
            "p_sonde_open":p_sonde_open, "sonde_open_mean":sonde_open_mean, "sonde_open_std":sonde_open_std,
            "p_sonde_strat":p_sonde_strat, "sonde_strat_mean":sonde_strat_mean, "sonde_strat_std":sonde_strat_std,
            "p_cam_open":p_cam_open, "cam_open_mean":cam_open_mean, "cam_open_std":cam_open_std,
            "p_cam_strat":p_cam_strat, "cam_strat_mean":cam_strat_mean, "cam_strat_std":cam_strat_std,
            "p_era_open":p_era_open, "era_open_mean":era_open_mean, "era_open_std":era_open_std,
            "p_era_strat":p_era_strat, "era_strat_mean":era_strat_mean, "era_strat_std":era_strat_std,
            "delta_open":delta_open, "delta_strat":delta_strat,
            "N_open":N_open, "N_strat":N_strat}
    for donde,donde_val in dropsinde_dict.items():
        print(f"type({donde}) = {type(donde_val)}")

    # save this case's own CAM profile (and its bias vs Obs) to disk for
    # later use (e.g. D3/browser); Obs/ERA5 are shared across cases and get
    # written once by the caller instead.
    save_named_profile_json(
        case, campaign, out_dir,
        p_cam_open, cam_open_mean, cam_open_std,
        p_cam_strat, cam_strat_mean, cam_strat_std,
        delta_open=delta_open, delta_strat=delta_strat,
    )

    return dropsinde_dict


def cam_obs_dropsonde_comp(adf, campaign, df_sonde, ds_outs, ds_era5):
    """
    
    """
    '''img_base = f'{campaign}__RHvprof_diff_coll__{cam_desc}'
    img_name = f'{img_base}.png'
    plot_loc = Path(adf.plot_location)
    img_path =  plot_loc / img_name
    if img_path.is_file():
        adf.add_website_data(str(img_path), img_base, campaign, plot_type="Dropsonde Comparison",
                            category="Model Cloud State", plot_loc=str(plot_loc),cam_desc=cam_desc)
    '''
    '''
    img_name = f'{img_base}.png'
    plot_loc = Path(adf.plot_location)
    img_path = f"{plot_loc}/{img_name}"

    if len(ds_outs.keys()) > 1:
        img_base = f'{campaign}__RHvprof_diff_coll__multi_CAM'
    else:
        case = list(ds_outs.keys())[0]
        img_base = f'{campaign}__RHvprof_diff_coll__{case}'
    if img_path.is_file():
        adf.add_website_data(str(img_path), img_base, campaign, plot_type="Dropsonde Comparison",
                            category="Model Cloud State", plot_loc=str(plot_loc),cam_desc="")
        return
    '''

    #Notify user that script has started:
    msg = "\n  Generating CAM-Obs Dropsonde plots..."
    print(f"{msg}\n  {'-' * (len(msg)-3)}")
    yup = Path(adf.config_file_dict["name"])
    plot_loc = Path(adf.plot_location)
    # pkl cache and JSON exports live alongside the dashboard/html files
    # (created already by AdfWeb.__init__, but mkdir here too to be safe):
    website_dir = plot_loc / "website"
    website_dir.mkdir(parents=True, exist_ok=True)
    # Qualified by campaign so a config listing multiple campaigns (e.g.
    # SOCRATES + CSET) doesn't have the second campaign's cases collide
    # with, or get mistaken for already-cached, the first's:
    data_pkl = website_dir / f"enough_{campaign}_{yup}.pkl"
    sonde_var = "rh" # tdry, u_wind, v_wind, rh
    cam_var = "RH" # T_K, RH, U, V
    era_var = "RH" # T_K, RH, U, V

    # safe_pickle_load tolerates a missing OR corrupted file (e.g. left
    # truncated by a previous run that got interrupted mid-write) by
    # just returning {}, so this one call covers "no cache yet" too:
    enough = cdog.safe_pickle_load(data_pkl) if Path(data_pkl).is_file() else {}

    # Check for any cases not already present in the cache - covers both
    # a brand new cache (every case missing) and one only partially
    # filled by a prior, interrupted run (just the remaining cases):
    missing_cases = [case for case in ds_outs if case not in enough]
    if missing_cases:
        print(f"Found new case(s) not in {data_pkl}: {missing_cases}")

        df_sonde["tdry"] = df_sonde["tdry"] + 273.15
        df_sonde["dp"]   = df_sonde["dp"]   + 273.15

        for case in missing_cases:
            enough[case] = compute_case_stats(
                case, campaign, website_dir, ds_outs[case], ds_era5, df_sonde, sonde_var, cam_var, era_var
            )

            # Persist after every case, not just once at the end, so an
            # interrupted run actually resumes from here next time
            # instead of silently redoing every case from scratch:
            cdog.atomic_pickle_dump(enough, data_pkl)

    # Obs/ERA5 are shared across cases, so write them once here rather than
    # once per case (mirrors the per-case CAM files compute_case_stats writes).
    first = next(iter(enough.values()))
    save_named_profile_json(
        "Obs", campaign, website_dir,
        first["p_sonde_open"], first["sonde_open_mean"], first["sonde_open_std"],
        first["p_sonde_strat"], first["sonde_strat_mean"], first["sonde_strat_std"],
        first["N_open"], first["N_strat"],
    )
    save_named_profile_json(
        "ERA5", campaign, website_dir,
        first["p_era_open"], first["era_open_mean"], first["era_open_std"],
        first["p_era_strat"], first["era_strat_mean"], first["era_strat_std"],
    )

    # Combined per-campaign JSON for the D3 dashboard (all cases in `enough`,
    # whichever branch above produced them):
    save_dropsonde_dashboard_json(campaign, sonde_var, enough, website_dir, yup)

    #=========================================
    # Figure 1. 3 panels with CAM, ERA5, Obs
    #=========================================
    #
    # fig = cdog.plot_three_panels_profiles_with_era(
    #     prof_sonde_open, prof_sonde_strat,
    #     p_cam_open,  cam_open_mean,  cam_open_std,
    #     p_cam_strat, cam_strat_mean, cam_strat_std,
    #     p_era_open,  era_open_mean,  era_open_std,
    #     p_era_strat, era_strat_mean, era_strat_std,
    #     N_open, N_strat,
    #     var_label="Relative Humidity (%)",
    #     ylim_hPa=(1000, 500), xlim_hPa=(-10,110),xlim_delta=(-35,20)
    # )
    # plt.savefig('SOCRATES_RHvprof_diff_coll_cam_era5_CAM6_6hrnudgUTV.png',dpi=300,bbox_inches='tight')

    #========================================================================
    # Figure 2. 3 panels with CAM and Obs, difference between cloud regimes
    #========================================================================
    # Skip building/saving plots entirely when the config has create_html
    # turned off - the pkl cache and dashboard JSON above are already
    # written regardless, so this just avoids unused PNGs and figure work.
    if not adf.create_html:
        print("  create_html is False: skipping dropsonde plot generation.")
        return

    plot_loc = Path(adf.plot_location)

    # Machine key -> (filename suffix used by plot_three_panels_cam_obs_delta_cam_minus_obs,
    # human-readable label for the dashboard's panel selector).
    panel_suffixes = {
        "open_cell":      ("OpenCell",      "Open-Cell"),
        "stratocumulus":  ("Stratocumulus", "Stratocumulus"),
        "open_cell_bias": ("OpenCellBias",  "Open-Cell Bias"),
        "stratocu_bias":  ("StratoCuBias",  "StratoCu Bias"),
    }

    def _make_and_register_suite(enough_subset, cam_desc, img_base):
        """
        Build one summary (4-panel) figure + its 4 individually-saved
        component panels for `enough_subset`, and register all 5 with the
        website generator under `cam_desc`.
        """
        fig = cdog.plot_three_panels_cam_obs_delta_cam_minus_obs(
                enough_subset,
                var_label="Relative Humidity (%)",
                ylim_hPa=(1000, 500), xlim_hPa=(-10,110), xlim_delta=(-20,35),
                save_indiv_panels=True,
                indiv_plot_loc=plot_loc,
                indiv_img_base=img_base,
        )

        img_name = f'{img_base}.png'
        img_path = plot_loc / img_name

        fig.savefig(img_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        adf.add_website_data(str(img_path), img_base, campaign, plot_type="Dropsonde Comparison",
                                category="Model Cloud State", plot_loc=str(plot_loc), cam_desc=cam_desc,
                                panel="summary", panel_label="Summary")

        for panel_key, (suffix, panel_label) in panel_suffixes.items():
            panel_img_base = f"{img_base}_{suffix}"
            panel_img_path = plot_loc / f"{panel_img_base}.png"
            adf.add_website_data(str(panel_img_path), panel_img_base, campaign, plot_type="Dropsonde Comparison",
                                    category="Model Cloud State", plot_loc=str(plot_loc), cam_desc=cam_desc,
                                    panel=panel_key, panel_label=panel_label)

    # With more than one case, build the "multi_CAM" ensemble suite first
    # (every case overlaid on the same axes vs Obs/ERA5), so it's the first
    # entry registered for this campaign and appears first in the dashboard's
    # case dropdown, ahead of the individual cases:
    if len(enough) > 1:
        img_base = f'{campaign}__RHvprof_diff_coll__multi_CAM'
        _make_and_register_suite(enough, "multi_CAM", img_base)

    # Every case also gets its own summary + component-panel suite (that
    # case alone vs Obs/ERA5), regardless of how many cases are in this run:
    for case in enough:
        img_base = f'{campaign}__RHvprof_diff_coll__{case}'
        _make_and_register_suite({case: enough[case]}, case, img_base)