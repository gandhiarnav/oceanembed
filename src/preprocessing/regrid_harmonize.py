"""
=============================================================================
OceanEmbed — Day 3: Spatial Harmonization & Multi-Source Regridding
=============================================================================
Purpose : Force all downloaded datasets (GLORYS, SMAP SSS, ERA5 Winds)
          onto the exact same 0.25° × 0.25° grid covering the North Indian Ocean.

Target Grid:
  Latitude  : 5.0°N  to 30.0°N  at 0.25° → 101 points
  Longitude : 45.0°E to 105.0°E at 0.25° → 241 points

Input  : data/raw/*.nc (GLORYS, SMAP SSS, ERA5 Winds)
Output : data/processed/surface_inputs_regridded.nc (Combined 7 surface channels)
         data/processed/target_temp_regridded.nc    (15 depth levels)
=============================================================================
"""

import os
import sys
import numpy as np
import xarray as xr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import xesmf as xe
    XESMF_AVAILABLE = True
except ImportError:
    XESMF_AVAILABLE = False
    print("  ℹ️  xESMF not installed. Using xarray linear interpolation.")

# ── Directory setup ───────────────────────────────────────────────────────────
RAW_DIR       = "./data/raw"
PROCESSED_DIR = "./data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Master Target Grid Definition ─────────────────────────────────────────────
LAT_TARGET = np.arange(5.0, 30.25, 0.25)    # 101 points
LON_TARGET = np.arange(45.0, 105.25, 0.25)  # 241 points
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

print("=" * 65)
print("  OceanEmbed — Spatial Harmonization & Multi-Source Regridding")
print("=" * 65)
print(f"  Target grid : {len(LAT_TARGET)} lat × {len(LON_TARGET)} lon (0.25° resolution)")
print(f"  Lat range   : {LAT_TARGET[0]}°N to {LAT_TARGET[-1]}°N")
print(f"  Lon range   : {LON_TARGET[0]}°E to {LON_TARGET[-1]}°E")
print()


def normalize_coord_names(ds: xr.Dataset) -> tuple[xr.Dataset, str, str]:
    """Find and normalize latitude and longitude dimension names."""
    lat_name = None
    lon_name = None
    
    for k in ds.coords:
        if k.lower() in ["lat", "latitude", "nav_lat"]:
            lat_name = k
        elif k.lower() in ["lon", "longitude", "nav_lon"]:
            lon_name = k
            
    if not lat_name:
        lat_name = "lat" if "lat" in ds.dims else "latitude"
    if not lon_name:
        lon_name = "lon" if "lon" in ds.dims else "longitude"
        
    return ds, lat_name, lon_name


def regrid_to_target(ds: xr.Dataset) -> xr.Dataset:
    """
    Regrid an xarray Dataset to the master 0.25° NIO grid (101x241).
    """
    ds, lat_name, lon_name = normalize_coord_names(ds)

    # Crop to NIO region with a 1-degree buffer for interpolation boundaries
    lat_slice = slice(4.0, 31.0)
    # Check if latitudes are descending (e.g. 90 to -90)
    if ds[lat_name].values[0] > ds[lat_name].values[-1]:
        lat_slice = slice(31.0, 4.0)

    ds_cropped = ds.sel({lat_name: lat_slice, lon_name: slice(44.0, 106.0)})

    if XESMF_AVAILABLE:
        ds_out = xr.Dataset({
            "lat": (["lat"], LAT_TARGET),
            "lon": (["lon"], LON_TARGET),
        })
        ds_in = ds_cropped.rename({lat_name: "lat", lon_name: "lon"})
        regridder = xe.Regridder(ds_in, ds_out, "bilinear",
                                 periodic=False, ignore_degenerate=True)
        ds_regridded = regridder(ds_in, keep_attrs=True)
    else:
        # Standard xarray linear interpolation
        ds_renamed = ds_cropped.rename({lat_name: "latitude", lon_name: "longitude"})
        ds_regridded = ds_renamed.interp(
            latitude=LAT_TARGET,
            longitude=LON_TARGET,
            method="linear"
        ).rename({"latitude": "lat", "longitude": "lon"})

    return ds_regridded


def align_time_coordinate(ds: xr.Dataset) -> xr.Dataset:
    """Ensure time coordinate is normalized to daily UTC midnight."""
    if "time" in ds.coords:
        # Floor timestamps to calendar days
        daily_times = ds["time"].dt.floor("D")
        ds = ds.assign_coords(time=daily_times)
    return ds


def validate_regridded_shape(ds: xr.Dataset, label: str) -> None:
    """Abort if the regridded dataset does not match the 101x241 contract."""
    n_lat = len(ds["lat"]) if "lat" in ds.dims else len(ds["latitude"])
    n_lon = len(ds["lon"]) if "lon" in ds.dims else len(ds["longitude"])
    if n_lat != 101 or n_lon != 241:
        print(f"\n  ❌ ABORT: Shape mismatch in '{label}'!")
        print(f"     Expected : lat=101, lon=241")
        print(f"     Got      : lat={n_lat}, lon={n_lon}")
        print("     Check LAT_TARGET / LON_TARGET ranges in this script.")
        sys.exit(1)
    print(f"       Grid check ✅ {label}: lat={n_lat}, lon={n_lon}")


# ── Collect and harmonize all surface datasets ────────────────────────────────
surface_datasets = []

# 1. GLORYS Surface Inputs (SST, SSH, U_curr, V_curr)
surface_file = os.path.join(RAW_DIR, "glorys_surface_inputs_jan2020.nc")
if os.path.exists(surface_file):
    print("  [1/4] Processing GLORYS surface inputs...")
    ds_surf = xr.open_dataset(surface_file)
    if "depth" in ds_surf.dims:
        ds_surf = ds_surf.squeeze("depth", drop=True)
    
    ds_surf_reg = regrid_to_target(ds_surf)
    ds_surf_reg = align_time_coordinate(ds_surf_reg)
    
    # Standardize variable names
    var_rename = {}
    if "thetao" in ds_surf_reg: var_rename["thetao"] = "sst"
    if "zos" in ds_surf_reg:    var_rename["zos"] = "ssh"
    if "uo" in ds_surf_reg:     var_rename["uo"] = "u_curr"
    if "vo" in ds_surf_reg:     var_rename["vo"] = "v_curr"
    ds_surf_reg = ds_surf_reg.rename(var_rename)
    
    surface_datasets.append(ds_surf_reg)
    print(f"        ✅ Regridded variables: {list(ds_surf_reg.data_vars)}")
else:
    print(f"  ⚠️  GLORYS surface file not found: {surface_file}")

# 2. NASA SMAP SSS
sss_file = os.path.join(RAW_DIR, "sss_merged_jan2020.nc")
if os.path.exists(sss_file):
    print("  [2/4] Processing NASA SMAP SSS...")
    ds_sss = xr.open_dataset(sss_file)
    ds_sss_reg = regrid_to_target(ds_sss)
    ds_sss_reg = align_time_coordinate(ds_sss_reg)
    
    if "sss_smap" in ds_sss_reg:
        ds_sss_reg = ds_sss_reg.rename({"sss_smap": "sss"})
    
    surface_datasets.append(ds_sss_reg)
    print(f"        ✅ Regridded variables: {list(ds_sss_reg.data_vars)}")
else:
    print(f"  ⚠️  SMAP SSS file not found: {sss_file}")

# 3. ERA5 Winds
winds_file = os.path.join(RAW_DIR, "era5_winds_jan2020.nc")
if os.path.exists(winds_file):
    print("  [3/4] Processing ERA5 10m surface winds...")
    ds_wind = xr.open_dataset(winds_file)
    ds_wind_reg = regrid_to_target(ds_wind)
    ds_wind_reg = align_time_coordinate(ds_wind_reg)
    
    var_rename = {}
    if "u10" in ds_wind_reg: var_rename["u10"] = "u_wind"
    if "v10" in ds_wind_reg: var_rename["v10"] = "v_wind"
    if var_rename:
        ds_wind_reg = ds_wind_reg.rename(var_rename)
        
    surface_datasets.append(ds_wind_reg)
    print(f"        ✅ Regridded variables: {list(ds_wind_reg.data_vars)}")
else:
    print(f"  ⚠️  ERA5 winds file not found: {winds_file}")

# ── Merge surface datasets into single unified NetCDF ────────────────────────
if surface_datasets:
    print("\n  Merging surface variables into single dataset...")
    # Merge along common dimensions (time, lat, lon)
    ds_surface_merged = xr.merge(surface_datasets, compat="override", join="inner")

    # ── Validate merged grid shape before saving ───────────────────────────
    validate_regridded_shape(ds_surface_merged, "surface_inputs_merged")

    out_surf_path = os.path.join(PROCESSED_DIR, "surface_inputs_regridded.nc")
    ds_surface_merged.to_netcdf(out_surf_path)
    print(f"  ✅ Saved unified surface file: {out_surf_path}")
    print(f"     Variables : {list(ds_surface_merged.data_vars)}")
    print(f"     Dimensions: {dict(ds_surface_merged.dims)}")
else:
    print("  ❌ No surface datasets available to merge.")

# ── Process GLORYS 3D Target ──────────────────────────────────────────────────
target_file = os.path.join(RAW_DIR, "glorys_target_temp_jan2020.nc")
if os.path.exists(target_file):
    print("\n  [4/4] Processing GLORYS 3D Target Temperature...")
    ds_tgt = xr.open_dataset(target_file)
    ds_tgt_reg = regrid_to_target(ds_tgt)
    ds_tgt_reg = align_time_coordinate(ds_tgt_reg)

    depth_dim = "depth" if "depth" in ds_tgt_reg.dims else "deptht"
    ds_tgt_depths = ds_tgt_reg.sel({depth_dim: STANDARD_DEPTHS}, method="nearest")

    # ── Validate target grid shape before saving ───────────────────────────
    validate_regridded_shape(ds_tgt_depths, "target_temp_regridded")
    n_depths = len(ds_tgt_depths[depth_dim])
    if n_depths != 15:
        print(f"\n  ❌ ABORT: Expected 15 depth levels, got {n_depths}")
        sys.exit(1)
    print(f"       Depth check ✅ target_temp: {n_depths} levels")

    out_tgt_path = os.path.join(PROCESSED_DIR, "target_temp_regridded.nc")
    ds_tgt_depths.to_netcdf(out_tgt_path)
    print(f"  ✅ Saved target file: {out_tgt_path}")
    print(f"     Depth levels: {n_depths} levels extracted")
    print(f"     Dimensions  : {dict(ds_tgt_depths.dims)}")
else:
    print(f"\n  ⚠️  GLORYS target file not found: {target_file}")

print("\n" + "=" * 65)
print("  ✅ Harmonization & Regridding complete!")
print("  Next step: Run python scripts/normalize.py")
print("=" * 65)
