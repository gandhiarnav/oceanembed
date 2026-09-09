"""
OceanEmbed — 3-Year Batch Data Extraction & Harmonization Pipeline (2018–2020)
Processes data month-by-month to prevent API timeouts and manage memory.
Exports final ML tensors (train/val/test) matching SIH-26066 specifications.

Usage:
    # Test a single month (e.g. 2018-01)
    python scripts/extract_3years.py --single-month 2018-01

    # Run full 3-year extraction (2018-01 to 2020-12)
    python scripts/extract_3years.py

    # Only merge existing processed monthly files and export final ML tensors
    python scripts/extract_3years.py --only-merge
"""
import os
import sys
import json
import argparse
import calendar
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
import numpy as np
import xarray as xr
from dotenv import load_dotenv

# Ensure project root in path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    TARGET_LAT,
    TARGET_LON,
    TARGET_DEPTHS,
    CANONICAL_CHANNELS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
)

load_dotenv(PROJECT_ROOT / ".env")

RAW_MONTHLY_DIR = RAW_DATA_DIR / "monthly"
PROCESSED_MONTHLY_DIR = PROCESSED_DATA_DIR / "monthly"


def get_month_days(year: int, month: int) -> int:
    """Return the number of days in the given year and month."""
    return calendar.monthrange(year, month)[1]


def normalize_coord_names(ds: xr.Dataset) -> Tuple[xr.Dataset, str, str]:
    """Find latitude and longitude coordinate names."""
    lat_name, lon_name = None, None
    for k in ds.coords:
        kl = k.lower()
        if kl in ["lat", "latitude", "nav_lat"]:
            lat_name = k
        elif kl in ["lon", "longitude", "nav_lon"]:
            lon_name = k
    if not lat_name:
        lat_name = "lat" if "lat" in ds.dims else "latitude"
    if not lon_name:
        lon_name = "lon" if "lon" in ds.dims else "longitude"
    return ds, lat_name, lon_name


def regrid_to_target_grid(
    ds: xr.Dataset,
    target_lat: np.ndarray = TARGET_LAT,
    target_lon: np.ndarray = TARGET_LON,
) -> xr.Dataset:
    """Regrid an xarray Dataset to the 0.25° NIO grid (101x241)."""
    ds, lat_name, lon_name = normalize_coord_names(ds)

    lat_vals = ds[lat_name].values
    is_lat_descending = len(lat_vals) > 1 and (lat_vals[0] > lat_vals[-1])
    lat_slice = slice(31.0, 4.0) if is_lat_descending else slice(4.0, 31.0)

    lon_vals = ds[lon_name].values
    is_lon_descending = len(lon_vals) > 1 and (lon_vals[0] > lon_vals[-1])
    lon_slice = slice(106.0, 44.0) if is_lon_descending else slice(44.0, 106.0)

    ds_cropped = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
    ds_renamed = ds_cropped.rename({lat_name: "latitude", lon_name: "longitude"})
    ds_regridded = ds_renamed.interp(
        latitude=target_lat,
        longitude=target_lon,
        method="linear"
    ).rename({"latitude": "lat", "longitude": "lon"})
    return ds_regridded


def align_time_to_midnight(ds: xr.Dataset) -> xr.Dataset:
    """Floor time coordinates to daily UTC midnight."""
    if "time" in ds.coords:
        daily_times = ds["time"].dt.floor("D")
        ds = ds.assign_coords(time=daily_times)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOWNLOAD FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def download_glorys_month(
    year: int,
    month: int,
    raw_dir: Path = RAW_MONTHLY_DIR,
) -> Tuple[Path, Path]:
    """
    Download GLORYS surface and 3D target for one month.
    Returns (raw_surface_path, raw_target_path).
    """
    import copernicusmarine

    user = os.getenv("COPERNICUSMARINE_SERVICE_USERNAME") or os.getenv("COPERNICUS_USERNAME")
    pwd = os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD") or os.getenv("COPERNICUS_PASSWORD")
    if not user or not pwd:
        raise ValueError("Copernicus Marine credentials missing in .env")

    raw_dir.mkdir(parents=True, exist_ok=True)
    num_days = get_month_days(year, month)
    start_iso = f"{year}-{month:02d}-01T00:00:00"
    end_iso = f"{year}-{month:02d}-{num_days:02d}T23:59:59"

    surf_fn = f"glorys_surf_{year}_{month:02d}.nc"
    tgt_fn = f"glorys_target_{year}_{month:02d}.nc"
    surf_path = raw_dir / surf_fn
    tgt_path = raw_dir / tgt_fn

    # 1. Surface inputs (SST, SSS, SSH, U/V currents)
    if not surf_path.exists() or surf_path.stat().st_size == 0:
        print(f"  [GLORYS Surface] Downloading {year}-{month:02d} ({start_iso} → {end_iso})...")
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            variables=["thetao", "so", "zos", "uo", "vo"],
            start_datetime=start_iso,
            end_datetime=end_iso,
            minimum_longitude=LON_MIN,
            maximum_longitude=LON_MAX,
            minimum_latitude=LAT_MIN,
            maximum_latitude=LAT_MAX,
            minimum_depth=0.0,
            maximum_depth=0.5,
            output_filename=surf_fn,
            output_directory=str(raw_dir),
            username=user,
            password=pwd,
            overwrite=True,
        )
        print(f"  ✅ GLORYS Surface downloaded: {surf_path} ({surf_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  ⏭️  GLORYS Surface exists: {surf_path}")

    # 2. 3D Target temperature (depth 0 to 1070m)
    if not tgt_path.exists() or tgt_path.stat().st_size == 0:
        print(f"  [GLORYS Target] Downloading {year}-{month:02d} ({start_iso} → {end_iso})...")
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            variables=["thetao"],
            start_datetime=start_iso,
            end_datetime=end_iso,
            minimum_longitude=LON_MIN,
            maximum_longitude=LON_MAX,
            minimum_latitude=LAT_MIN,
            maximum_latitude=LAT_MAX,
            minimum_depth=0.0,
            maximum_depth=1070.0,
            output_filename=tgt_fn,
            output_directory=str(raw_dir),
            username=user,
            password=pwd,
            overwrite=True,
        )
        print(f"  ✅ GLORYS Target downloaded: {tgt_path} ({tgt_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  ⏭️  GLORYS Target exists: {tgt_path}")

    return surf_path, tgt_path


def download_era5_month(
    year: int,
    month: int,
    raw_dir: Path = RAW_MONTHLY_DIR,
) -> Path:
    """
    Download and daily-average ERA5 10m surface winds (u10, v10) for one month.
    """
    import cdsapi

    raw_dir.mkdir(parents=True, exist_ok=True)
    num_days = get_month_days(year, month)
    out_fn = f"era5_winds_{year}_{month:02d}.nc"
    out_path = raw_dir / out_fn
    tmp_hourly = raw_dir / f"tmp_hourly_{out_fn}"

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  ⏭️  ERA5 Winds exist: {out_path}")
        return out_path

    print(f"  [ERA5 Winds] Downloading {year}-{month:02d} from CDS...")
    cds_url = os.getenv("CDSAPI_URL")
    cds_key = os.getenv("CDSAPI_KEY")

    if cds_url and cds_key and not cds_key.startswith("your-"):
        client = cdsapi.Client(url=cds_url, key=cds_key)
    else:
        client = cdsapi.Client()

    days = [f"{d:02d}" for d in range(1, num_days + 1)]
    hours = ["00:00", "06:00", "12:00", "18:00"]

    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year": str(year),
            "month": f"{month:02d}",
            "day": days,
            "time": hours,
            "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],  # [N, W, S, E]
        },
        str(tmp_hourly),
    )

    # Compute daily average
    ds = xr.open_dataset(tmp_hourly)
    time_name = "valid_time" if "valid_time" in ds.coords else ("time" if "time" in ds.coords else None)
    if time_name != "time" and time_name is not None:
        ds = ds.rename({time_name: "time"})

    ds_daily = ds.resample(time="1D").mean()
    if "time" in ds_daily.coords:
        ds_daily["time"].encoding.clear()

    ds_daily.to_netcdf(out_path)
    ds.close()
    if tmp_hourly.exists():
        tmp_hourly.unlink()

    print(f"  ✅ ERA5 Daily Winds saved: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 2. REGRIDDING & HARMONIZATION PER MONTH
# ─────────────────────────────────────────────────────────────────────────────

def regrid_and_save_month(
    year: int,
    month: int,
    raw_surf_path: Path,
    raw_tgt_path: Path,
    raw_era5_path: Path,
    processed_monthly_dir: Path = PROCESSED_MONTHLY_DIR,
) -> Tuple[Path, Path]:
    """
    Regrids surface inputs and target temperature for one month to 0.25° (101x241).
    Interpolates target vertically to 15 TARGET_DEPTHS.
    Returns (processed_surf_path, processed_tgt_path).
    """
    processed_monthly_dir.mkdir(parents=True, exist_ok=True)
    out_surf_path = processed_monthly_dir / f"surface_{year}_{month:02d}.nc"
    out_tgt_path = processed_monthly_dir / f"target_{year}_{month:02d}.nc"

    num_days = get_month_days(year, month)

    # 1. Regrid Surface Inputs
    if not out_surf_path.exists() or out_surf_path.stat().st_size == 0:
        print(f"  [Harmonize Surface] Regridding surface variables for {year}-{month:02d}...")
        ds_surf = xr.open_dataset(raw_surf_path)
        if "depth" in ds_surf.dims:
            ds_surf = ds_surf.squeeze("depth", drop=True)
        ds_surf_reg = regrid_to_target_grid(ds_surf)
        ds_surf_reg = align_time_to_midnight(ds_surf_reg)

        # Rename GLORYS variables
        var_map = {}
        if "thetao" in ds_surf_reg: var_map["thetao"] = "sst"
        if "so" in ds_surf_reg: var_map["so"] = "sss"
        if "zos" in ds_surf_reg: var_map["zos"] = "ssh"
        if "uo" in ds_surf_reg: var_map["uo"] = "u_curr"
        if "vo" in ds_surf_reg: var_map["vo"] = "v_curr"
        if var_map:
            ds_surf_reg = ds_surf_reg.rename(var_map)

        # Regrid ERA5 Winds
        ds_era5 = xr.open_dataset(raw_era5_path)
        ds_era5_reg = regrid_to_target_grid(ds_era5)
        ds_era5_reg = align_time_to_midnight(ds_era5_reg)

        wind_map = {}
        if "u10" in ds_era5_reg: wind_map["u10"] = "u_wind"
        if "v10" in ds_era5_reg: wind_map["v10"] = "v_wind"
        if wind_map:
            ds_era5_reg = ds_era5_reg.rename(wind_map)

        # Merge surface datasets
        keep_surf = [v for v in ["sst", "sss", "ssh", "u_curr", "v_curr"] if v in ds_surf_reg]
        keep_wind = [v for v in ["u_wind", "v_wind"] if v in ds_era5_reg]

        merged_surf = xr.merge([ds_surf_reg[keep_surf], ds_era5_reg[keep_wind]], join="inner")

        # Verify shapes
        assert len(merged_surf["lat"]) == len(TARGET_LAT), f"Lat mismatch: {len(merged_surf['lat'])}"
        assert len(merged_surf["lon"]) == len(TARGET_LON), f"Lon mismatch: {len(merged_surf['lon'])}"

        if "time" in merged_surf.coords:
            merged_surf["time"].encoding.clear()

        merged_surf.to_netcdf(out_surf_path)
        ds_surf.close()
        ds_era5.close()
        print(f"  ✅ Saved regridded surface: {out_surf_path} ({out_surf_path.stat().st_size / 1e6:.1f} MB, {len(merged_surf.time)} days)")
    else:
        print(f"  ⏭️  Regridded surface exists: {out_surf_path}")

    # 2. Regrid Target Temperature (15 depths)
    if not out_tgt_path.exists() or out_tgt_path.stat().st_size == 0:
        print(f"  [Harmonize Target] Regridding 3D target for {year}-{month:02d}...")
        ds_tgt = xr.open_dataset(raw_tgt_path)
        depth_dim = "depth" if "depth" in ds_tgt.dims else "deptht"

        # Interpolate vertically
        native_depths = ds_tgt[depth_dim].values
        if native_depths[0] > 0.0:
            surface_slice = ds_tgt.isel({depth_dim: 0}).assign_coords({depth_dim: 0.0})
            ds_tgt_padded = xr.concat([surface_slice, ds_tgt], dim=depth_dim)
        else:
            ds_tgt_padded = ds_tgt

        ds_tgt_interp = ds_tgt_padded.interp({depth_dim: TARGET_DEPTHS}, method="linear")
        ds_tgt_reg = regrid_to_target_grid(ds_tgt_interp)
        ds_tgt_reg = align_time_to_midnight(ds_tgt_reg)
        ds_tgt_reg = ds_tgt_reg.transpose("time", depth_dim, "lat", "lon")

        assert len(ds_tgt_reg[depth_dim]) == len(TARGET_DEPTHS), f"Depth levels mismatch"
        assert len(ds_tgt_reg["lat"]) == len(TARGET_LAT)
        assert len(ds_tgt_reg["lon"]) == len(TARGET_LON)

        if "time" in ds_tgt_reg.coords:
            ds_tgt_reg["time"].encoding.clear()

        ds_tgt_reg.to_netcdf(out_tgt_path)
        ds_tgt.close()
        print(f"  ✅ Saved regridded target: {out_tgt_path} ({out_tgt_path.stat().st_size / 1e6:.1f} MB, 15 depths)")
    else:
        print(f"  ⏭️  Regridded target exists: {out_tgt_path}")

    return out_surf_path, out_tgt_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. MERGING & NORMALIZATION ACROSS ALL MONTHS
# ─────────────────────────────────────────────────────────────────────────────

def merge_all_months_and_normalize(
    processed_monthly_dir: Path = PROCESSED_MONTHLY_DIR,
    out_dir: Path = PROCESSED_DATA_DIR,
    train_end_date: str = "2019-12-31",
    val_end_date: str = "2020-06-30",
):
    """
    Concatenates all monthly NetCDF files into unified 3-year tensors,
    splits into train/val/test, computes leakage-free normalization statistics,
    and exports final numpy tensors and metadata.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    surf_files = sorted(list(processed_monthly_dir.glob("surface_*.nc")))
    tgt_files = sorted(list(processed_monthly_dir.glob("target_*.nc")))

    if not surf_files or not tgt_files:
        raise FileNotFoundError(f"No processed monthly files found in {processed_monthly_dir}")

    print("\n" + "=" * 65)
    print(f"  Merging {len(surf_files)} Surface Months & {len(tgt_files)} Target Months...")
    print("=" * 65)

    ds_surf = xr.open_mfdataset(surf_files, combine="by_coords")
    ds_tgt = xr.open_mfdataset(tgt_files, combine="by_coords")

    # Ensure temporal alignment
    common_times = np.intersect1d(ds_surf.time.values, ds_tgt.time.values)
    ds_surf = ds_surf.sel(time=common_times)
    ds_tgt = ds_tgt.sel(time=common_times)
    n_times = len(common_times)

    print(f"  Total aligned days: {n_times}")
    print(f"  Date range        : {str(common_times[0])[:10]} → {str(common_times[-1])[:10]}")

    # Build Master Ocean Mask
    tgt_var = "thetao" if "thetao" in ds_tgt else list(ds_tgt.data_vars)[0]
    depth_dim = "depth" if "depth" in ds_tgt.dims else "deptht"
    sample_surface_tgt = ds_tgt[tgt_var].isel(time=0).sel({depth_dim: 0.0}, method="nearest").values
    ocean_mask = ~np.isnan(sample_surface_tgt)
    np.save(out_dir / "ocean_mask.npy", ocean_mask)
    print(f"  Saved ocean_mask.npy: {ocean_mask.sum()} ocean pixels ({ocean_mask.mean()*100:.1f}%)")

    # Map 7 canonical input channels
    channel_order = ["sst", "sss", "ssh", "u_curr", "v_curr", "u_wind", "v_wind"]
    for ch in channel_order:
        if ch not in ds_surf.data_vars:
            raise ValueError(f"Required channel '{ch}' missing from surface dataset!")

    n_lat = len(TARGET_LAT)
    n_lon = len(TARGET_LON)
    n_chan = len(channel_order)

    print("  Loading all surface inputs into memory...")
    raw_inputs = np.zeros((n_times, n_chan, n_lat, n_lon), dtype=np.float32)
    for idx, ch in enumerate(channel_order):
        raw_inputs[:, idx, :, :] = ds_surf[ch].values.astype(np.float32)

    print("  Loading all target temperatures into memory...")
    raw_targets = ds_tgt[tgt_var].values.astype(np.float32)

    # Temporal splits: Train (2018-2019), Val (H1 2020), Test (H2 2020)
    time_strs = [str(t)[:10] for t in common_times]
    dt_train_end = np.datetime64(train_end_date)
    dt_val_end = np.datetime64(val_end_date)

    train_mask = common_times <= dt_train_end
    val_mask = (common_times > dt_train_end) & (common_times <= dt_val_end)
    test_mask = common_times > dt_val_end

    splits = {
        "train": np.where(train_mask)[0],
        "val": np.where(val_mask)[0],
        "test": np.where(test_mask)[0],
    }

    print("\n  Temporal Splits:")
    for s_name, indices in splits.items():
        if len(indices) > 0:
            print(f"    {s_name:5s}: {len(indices):4d} days ({time_strs[indices[0]]} → {time_strs[indices[-1]]})")
        else:
            print(f"    {s_name:5s}:    0 days")

    # Compute Normalization Stats ONLY on Training Split
    print("\n  Computing normalization statistics (strictly on training split)...")
    train_idx = splits["train"]
    if len(train_idx) == 0:
        raise ValueError("Train split is empty! Check date ranges.")

    train_inputs = raw_inputs[train_idx]
    norm_stats = {}
    for idx, ch_name in enumerate(channel_order):
        ch_ocean = train_inputs[:, idx, ocean_mask]
        mean_v = float(np.nanmean(ch_ocean))
        std_v = float(np.nanstd(ch_ocean))
        if std_v < 1e-8 or np.isnan(std_v):
            std_v = 1.0
        norm_stats[ch_name] = {"mean": mean_v, "std": std_v}
        print(f"    {ch_name:8s}: mean = {mean_v:8.4f}, std = {std_v:8.4f}")

    with open(out_dir / "norm_stats.json", "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Saved norm_stats.json")

    # Normalize & Export Split Tensors
    split_info = {}
    for s_name, indices in splits.items():
        if len(indices) == 0:
            continue
        s_in = raw_inputs[indices].copy()
        s_tgt = raw_targets[indices].copy()

        # Normalize inputs with train stats
        for idx, ch_name in enumerate(channel_order):
            mv = norm_stats[ch_name]["mean"]
            sv = norm_stats[ch_name]["std"]
            s_in[:, idx, :, :] = (s_in[:, idx, :, :] - mv) / sv
            s_in[:, idx, ~ocean_mask] = 0.0
            s_in[:, idx, :, :] = np.nan_to_num(s_in[:, idx, :, :], nan=0.0)

        # Land NaNs in targets set to 0.0
        s_tgt = np.nan_to_num(s_tgt, nan=0.0)

        in_file = out_dir / f"{s_name}_inputs.npy"
        tgt_file = out_dir / f"{s_name}_targets.npy"
        np.save(in_file, s_in)
        np.save(tgt_file, s_tgt)

        split_info[s_name] = {
            "num_timesteps": int(len(indices)),
            "date_range": [time_strs[indices[0]], time_strs[indices[-1]]],
            "input_shape": list(s_in.shape),
            "target_shape": list(s_tgt.shape),
        }
        print(f"  ✅ Saved {s_name} inputs : {in_file.name} shape={s_in.shape} dtype={s_in.dtype}")
        print(f"  ✅ Saved {s_name} targets: {tgt_file.name} shape={s_tgt.shape} dtype={s_tgt.dtype}")

    with open(out_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"  Saved split_info.json")

    # Verification
    print("\n" + "=" * 65)
    print("  VERIFICATION:")
    print("=" * 65)
    for s_name in split_info:
        in_arr = np.load(out_dir / f"{s_name}_inputs.npy")
        tgt_arr = np.load(out_dir / f"{s_name}_targets.npy")
        assert np.isnan(in_arr).sum() == 0, f"NaNs found in {s_name} inputs"
        assert np.isnan(tgt_arr).sum() == 0, f"NaNs found in {s_name} targets"
        print(f"  ✅ {s_name:5s}: Inputs={in_arr.shape}, Targets={tgt_arr.shape}, NaNs=0")

    print("\n" + "=" * 65)
    print("  🎉 3-YEAR ML DATASET EXTRACTION COMPLETE!")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN BATCH ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    start_year: int = 2018,
    end_year: int = 2020,
    single_month: Optional[str] = None,
    only_merge: bool = False,
    cleanup_raw: bool = False,
):
    print("=" * 65)
    print("  OceanEmbed — 3-Year Batch Extraction & Harmonization")
    print("=" * 65)

    if only_merge:
        print("  Mode: ONLY MERGE existing monthly files")
        merge_all_months_and_normalize()
        return

    # Build list of (year, month) to process
    if single_month:
        sy, sm = map(int, single_month.split("-"))
        months_to_run = [(sy, sm)]
        print(f"  Mode: Single month test [{sy}-{sm:02d}]")
    else:
        months_to_run = []
        for y in range(start_year, end_year + 1):
            for m in range(1, 13):
                months_to_run.append((y, m))
        print(f"  Mode: Full multi-year run ({len(months_to_run)} months: {start_year}-01 → {end_year}-12)")

    total_months = len(months_to_run)
    for idx, (yr, mo) in enumerate(months_to_run, 1):
        month_label = f"{yr}-{mo:02d}"
        print(f"\n[{idx}/{total_months}] ═══ Processing Month: {month_label} ═══")

        # Check if monthly regridded outputs already exist
        surf_out = PROCESSED_MONTHLY_DIR / f"surface_{yr}_{mo:02d}.nc"
        tgt_out = PROCESSED_MONTHLY_DIR / f"target_{yr}_{mo:02d}.nc"

        if surf_out.exists() and tgt_out.exists() and surf_out.stat().st_size > 0 and tgt_out.stat().st_size > 0:
            print(f"  ⏩ Month {month_label} already processed and regridded! Skipping.")
            continue

        try:
            # 1. Download
            raw_surf, raw_tgt = download_glorys_month(yr, mo)
            raw_era5 = download_era5_month(yr, mo)

            # 2. Regrid and harmonize
            regrid_and_save_month(yr, mo, raw_surf, raw_tgt, raw_era5)

            # 3. Optional cleanup of raw month files
            if cleanup_raw:
                for rf in [raw_surf, raw_tgt, raw_era5]:
                    if rf.exists():
                        rf.unlink()
                print(f"  🧹 Cleaned up raw files for {month_label}")

        except Exception as e:
            print(f"\n  ❌ Failed processing month {month_label}: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # If full run or multiple months, merge and export
    if not single_month:
        merge_all_months_and_normalize()
    else:
        sm_str = single_month.replace("-", "_")
        print(f"\n  ✅ Single month test {single_month} completed successfully!")
        print(f"     Regridded surface: {PROCESSED_MONTHLY_DIR / f'surface_{sm_str}.nc'}")
        print(f"     Regridded target : {PROCESSED_MONTHLY_DIR / f'target_{sm_str}.nc'}")


def main():
    parser = argparse.ArgumentParser(description="Extract and harmonize 3-year ocean dataset.")
    parser.add_argument("--start-year", type=int, default=2018, help="Start year (default: 2018)")
    parser.add_argument("--end-year", type=int, default=2020, help="End year (default: 2020)")
    parser.add_argument("--single-month", type=str, default=None, help="Test a single month (format: YYYY-MM)")
    parser.add_argument("--only-merge", action="store_true", help="Only merge existing monthly files")
    parser.add_argument("--cleanup-raw", action="store_true", help="Delete raw month files after regridding to save space")

    args = parser.parse_args()
    run_pipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        single_month=args.single_month,
        only_merge=args.only_merge,
        cleanup_raw=args.cleanup_raw,
    )


if __name__ == "__main__":
    main()
