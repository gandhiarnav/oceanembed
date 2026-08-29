"""
=============================================================================
OceanEmbed — Day 6: ARGO Float Co-location Script
=============================================================================
Purpose : Match raw ARGO float profiles (sparse lat/lon/time points)
          to the nearest model grid cell and date, then interpolate
          the ARGO temperatures to the 15 standard depth levels.

This produces an apples-to-apples comparison between:
  - ARGO real measurements  (truth)
  - GLORYS reanalysis       (what the model was trained on)
  - Model predictions       (what we are validating)

Input  : Raw ARGO .nc file (from Argo GDAC or INCOIS)
Output : data/validation/argo_collocated.csv

CSV Columns:
  date, float_id, argo_lat, argo_lon,
  grid_lat, grid_lon,
  depth_m,
  argo_temp, glorys_temp, model_prediction

Usage:
  python validation/collocate_argo.py --argo-file data/raw/argo_profiles.nc
=============================================================================
"""

import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import interp1d
import os
import argparse

# ── Standard depth levels (must match the model output exactly) ───────────────
STANDARD_DEPTHS = np.array([0, 5, 10, 20, 30, 50, 75, 100,
                             125, 150, 200, 300, 500, 700, 1000])

# ── Grid definition ───────────────────────────────────────────────────────────
LAT_GRID = np.arange(5.0, 30.25, 0.25)    # 101 points
LON_GRID = np.arange(45.0, 105.25, 0.25)  # 241 points

OUTPUT_DIR = "./data/validation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_nearest_grid_point(argo_lat: float, argo_lon: float):
    """
    Find the nearest 0.25° grid point for a given ARGO float position.
    Returns (lat_idx, lon_idx, grid_lat, grid_lon).
    """
    lat_idx = int(np.argmin(np.abs(LAT_GRID - argo_lat)))
    lon_idx = int(np.argmin(np.abs(LON_GRID - argo_lon)))
    return lat_idx, lon_idx, LAT_GRID[lat_idx], LON_GRID[lon_idx]


def interpolate_to_standard_depths(argo_depths: np.ndarray,
                                   argo_temps: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate ARGO temperature profile to the 15 standard depths.
    Values outside the ARGO measurement range are extrapolated (or set to NaN).
    """
    # Remove NaN values from ARGO profile
    valid = ~(np.isnan(argo_depths) | np.isnan(argo_temps))
    if valid.sum() < 2:
        return np.full(len(STANDARD_DEPTHS), np.nan)

    f = interp1d(
        argo_depths[valid], argo_temps[valid],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan   # NaN outside measurement range (don't extrapolate)
    )
    return f(STANDARD_DEPTHS)


def collocate_profiles(argo_ds: xr.Dataset,
                       glorys_ds: xr.Dataset = None) -> pd.DataFrame:
    """
    Co-locate all ARGO profiles against the model grid.

    Parameters
    ----------
    argo_ds   : xarray Dataset of ARGO profiles
                Must have variables: PSAL_ADJUSTED or PSAL, TEMP_ADJUSTED or TEMP,
                and dimensions N_PROF, N_LEVELS
    glorys_ds : Optional — regridded GLORYS for extracting nearest grid temperature

    Returns
    -------
    DataFrame with one row per (profile, depth_level)
    """
    records = []

    # Detect ARGO variable names (files differ slightly)
    temp_var  = "TEMP_ADJUSTED" if "TEMP_ADJUSTED" in argo_ds else "TEMP"
    depth_var = "PRES_ADJUSTED" if "PRES_ADJUSTED" in argo_ds else "PRES"
    lat_var   = "LATITUDE"
    lon_var   = "LONGITUDE"
    time_var  = "JULD"

    n_profiles = argo_ds.dims.get("N_PROF", 0)
    print(f"  Processing {n_profiles} ARGO profiles ...")

    for p_idx in range(n_profiles):
        argo_lat  = float(argo_ds[lat_var].values[p_idx])
        argo_lon  = float(argo_ds[lon_var].values[p_idx])
        argo_date = pd.Timestamp(argo_ds[time_var].values[p_idx])

        # Skip profiles outside NIO bounding box
        if not (5.0 <= argo_lat <= 30.0 and 45.0 <= argo_lon <= 105.0):
            continue

        # Find nearest grid point
        lat_idx, lon_idx, g_lat, g_lon = find_nearest_grid_point(argo_lat, argo_lon)

        # Get ARGO depths and temperatures for this profile
        depths = argo_ds[depth_var].values[p_idx, :]  # pressure ≈ depth in m
        temps  = argo_ds[temp_var].values[p_idx, :]

        # Interpolate to standard depth levels
        temps_standard = interpolate_to_standard_depths(depths, temps)

        # Extract GLORYS temperature at this grid point and date (if available)
        glorys_temps = np.full(len(STANDARD_DEPTHS), np.nan)
        if glorys_ds is not None:
            try:
                glorys_profile = glorys_ds["thetao"].sel(
                    time=argo_date, method="nearest"
                ).isel(lat=lat_idx, lon=lon_idx).values
                glorys_temps = glorys_profile
            except Exception:
                pass  # Leave as NaN if extraction fails

        # Store one record per depth level
        float_id = str(argo_ds.get("PLATFORM_NUMBER", {}).values[p_idx]).strip() \
                   if "PLATFORM_NUMBER" in argo_ds else f"profile_{p_idx}"

        for d_idx, depth in enumerate(STANDARD_DEPTHS):
            records.append({
                "date"            : argo_date.strftime("%Y-%m-%d"),
                "float_id"        : float_id,
                "argo_lat"        : round(argo_lat, 3),
                "argo_lon"        : round(argo_lon, 3),
                "grid_lat"        : g_lat,
                "grid_lon"        : g_lon,
                "depth_m"         : depth,
                "argo_temp"       : round(float(temps_standard[d_idx]), 4)
                                    if not np.isnan(temps_standard[d_idx]) else None,
                "glorys_temp"     : round(float(glorys_temps[d_idx]), 4)
                                    if not np.isnan(glorys_temps[d_idx]) else None,
                "model_prediction": None,  # Filled in by Person 2 after training
            })

        if p_idx % 100 == 0:
            print(f"    Processed {p_idx + 1}/{n_profiles} profiles ...")

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARGO Co-location Script")
    parser.add_argument("--argo-file",   type=str, required=True,
                        help="Path to ARGO NetCDF file")
    parser.add_argument("--glorys-file", type=str, default=None,
                        help="Path to regridded GLORYS NetCDF file (optional)")
    args = parser.parse_args()

    print("=" * 65)
    print("  OceanEmbed — ARGO Float Co-location")
    print("=" * 65)

    if not os.path.exists(args.argo_file):
        print(f"  ❌ ARGO file not found: {args.argo_file}")
        exit(1)

    print(f"  Loading ARGO data: {args.argo_file}")
    argo_ds = xr.open_dataset(args.argo_file)

    glorys_ds = None
    if args.glorys_file and os.path.exists(args.glorys_file):
        print(f"  Loading GLORYS data: {args.glorys_file}")
        glorys_ds = xr.open_dataset(args.glorys_file)

    df = collocate_profiles(argo_ds, glorys_ds)

    out_path = os.path.join(OUTPUT_DIR, "argo_collocated.csv")
    df.to_csv(out_path, index=False)

    print(f"\n  ✅ Co-location complete!")
    print(f"     Total rows   : {len(df)}")
    print(f"     Valid profiles: {df['argo_temp'].notna().sum()} depth readings")
    print(f"     Saved CSV    : {out_path}")
    print()
    print("  Hand this CSV to Person 2 to fill in 'model_prediction' column")
    print("  and compute final RMSE/Bias/R² metrics.")
