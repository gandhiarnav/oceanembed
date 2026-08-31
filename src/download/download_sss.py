"""
OceanEmbed — Download Sea Surface Salinity (SSS) — NASA SMAP L4 (RSS, 0.25°)
Dataset: SMAP_RSS_L4_SSS_8DAY_RUNNINGMEAN_V6 (or V5 fallback)
Variable: sss_smap (PSU)
"""
import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import xarray as xr
import earthaccess
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RAW_DATA_DIR, RAW_SSS_DIR

load_dotenv()

SMAP_SHORT_NAME = "SMAP_RSS_L3_SSS_SMI_8DAY-RUNNINGMEAN_V6"
SMAP_VAR = "sss_smap"


def download_smap_sss(
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-31",
    output_filename: str = "sss_merged_jan2020.nc",
    raw_dir: Path = RAW_SSS_DIR,
    output_dir: Path = RAW_DATA_DIR
):
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Downloading SMAP L4 Sea Surface Salinity (RSS, 0.25°)")
    print("=" * 65)
    print(f"  Product   : {SMAP_SHORT_NAME}")
    print(f"  Variable  : {SMAP_VAR} (PSU)")
    print(f"  Region    : {LAT_MIN}°N–{LAT_MAX}°N, {LON_MIN}°E–{LON_MAX}°E")
    print(f"  Period    : {start_date} → {end_date}")
    print(f"  Raw files : {raw_dir}/")
    print(f"  Merged    : {output_dir / output_filename}")
    print()

    # Step 1: Login
    print("  [1/4] Authenticating with NASA Earthdata...")
    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        print("❌ NASA Earthdata authentication failed.")
        print("   Check EARTHDATA_USERNAME and EARTHDATA_PASSWORD in .env")
        sys.exit(1)
    print(f"  ✅ Logged in as: {auth.username}\n")

    # Step 2: Search
    print(f"  [2/4] Searching for SMAP granules ({start_date} → {end_date})...")
    results = earthaccess.search_data(
        short_name=SMAP_SHORT_NAME,
        temporal=(start_date, end_date),
        bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
    )
    print(f"  ✅ Found {len(results)} granule(s)\n")

    if len(results) == 0:
        print(f"  ⚠️  No granules found for {SMAP_SHORT_NAME} in {start_date} → {end_date}.")
        return None

    # Step 3: Download
    print(f"  [3/4] Downloading {len(results)} granule(s) → {raw_dir}/")
    downloaded_files = earthaccess.download(results, local_path=str(raw_dir))
    print(f"\n  ✅ Downloaded {len(downloaded_files)} file(s)\n")

    # Step 4: Crop & Merge
    print("  [4/4] Cropping to NIO box and merging into single NetCDF...")
    datasets = []
    for fpath in sorted(downloaded_files):
        try:
            ds = xr.open_dataset(fpath)
            lat_name = "lat" if "lat" in ds.coords else "latitude"
            lon_name = "lon" if "lon" in ds.coords else "longitude"

            ds_crop = ds.sel(
                {lat_name: slice(LAT_MIN, LAT_MAX), lon_name: slice(LON_MIN, LON_MAX)}
            )

            if SMAP_VAR in ds_crop.data_vars:
                ds_crop = ds_crop[[SMAP_VAR]].rename({SMAP_VAR: "sss"})
            else:
                avail = list(ds_crop.data_vars)
                print(f"  ⚠️  '{SMAP_VAR}' not found. Using '{avail[0]}'")
                ds_crop = ds_crop[[avail[0]]].rename({avail[0]: "sss"})

            if "time" not in ds_crop["sss"].dims and "time" in ds.coords:
                ds_crop["sss"] = ds_crop["sss"].expand_dims({"time": ds["time"].values})

            datasets.append(ds_crop)
        except Exception as e:
            print(f"  ⚠️  Could not process {fpath}: {e}")

    if not datasets:
        print("❌ No datasets could be merged.")
        return None

    # Merge and drop duplicate time steps if any
    ds_merged = xr.concat(datasets, dim="time")
    _, index = np.unique(ds_merged["time"], return_index=True)
    ds_merged = ds_merged.isel(time=index).sortby("time")
    if "time" in ds_merged.coords:
        ds_merged["time"].encoding.clear()

    merged_path = output_dir / output_filename
    ds_merged.to_netcdf(merged_path)
    print(f"  ✅ Merged file written: {merged_path}\n")

    # Quick check
    sss_vals = ds_merged["sss"].values
    ocean_vals = sss_vals[~np.isnan(sss_vals)]
    print(f"  ✅ Salinity range : {ocean_vals.min():.2f} – {ocean_vals.max():.2f} PSU")
    print(f"  ✅ NaN fraction   : {float(np.isnan(sss_vals).mean()) * 100:.1f}%")
    print(f"  ✅ SSS Download COMPLETE: {merged_path}")
    return merged_path


def main():
    parser = argparse.ArgumentParser(description="Download SMAP SSS data.")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2020-01-31", help="End date YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="sss_merged_jan2020.nc", help="Output filename")

    args = parser.parse_args()
    download_smap_sss(start_date=args.start, end_date=args.end, output_filename=args.output)


if __name__ == "__main__":
    main()
