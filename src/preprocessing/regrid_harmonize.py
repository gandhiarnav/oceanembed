"""
OceanEmbed — Spatial Harmonization & Multi-Source Regridding Pipeline.
Harmonizes multi-source raw ocean datasets (GLORYS, CMC SST, SMAP SSS, ERA5 Winds)
onto the canonical 0.25° × 0.25° grid covering the North Indian Ocean (101 x 241).
"""
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
import xarray as xr

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TARGET_LAT,
    TARGET_LON,
    TARGET_DEPTHS,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_SST_DIR,
    CANONICAL_CHANNELS,
)

try:
    import xesmf as xe
    XESMF_AVAILABLE = True
except ImportError:
    XESMF_AVAILABLE = False


def normalize_coord_names(ds: xr.Dataset) -> Tuple[xr.Dataset, str, str]:
    """Find and return latitude and longitude coordinate/dimension names."""
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


def regrid_to_target(
    ds: xr.Dataset,
    target_lat: np.ndarray = TARGET_LAT,
    target_lon: np.ndarray = TARGET_LON,
    method: str = "linear"
) -> xr.Dataset:
    """
    Regrid an xarray Dataset to the master 0.25° NIO grid (101x241).
    """
    ds, lat_name, lon_name = normalize_coord_names(ds)

    # Crop to NIO region with a 1-degree buffer for interpolation boundaries
    lat_vals = ds[lat_name].values
    is_lat_descending = len(lat_vals) > 1 and (lat_vals[0] > lat_vals[-1])
    lat_slice = slice(31.0, 4.0) if is_lat_descending else slice(4.0, 31.0)

    lon_vals = ds[lon_name].values
    is_lon_descending = len(lon_vals) > 1 and (lon_vals[0] > lon_vals[-1])
    lon_slice = slice(106.0, 44.0) if is_lon_descending else slice(44.0, 106.0)

    ds_cropped = ds.sel({lat_name: lat_slice, lon_name: lon_slice})

    if XESMF_AVAILABLE and method == "xesmf_bilinear":
        ds_out = xr.Dataset({
            "lat": (["lat"], target_lat),
            "lon": (["lon"], target_lon),
        })
        ds_in = ds_cropped.rename({lat_name: "lat", lon_name: "lon"})
        regridder = xe.Regridder(ds_in, ds_out, "bilinear", periodic=False, ignore_degenerate=True)
        ds_regridded = regridder(ds_in, keep_attrs=True)
    else:
        # Standard xarray linear interpolation
        ds_renamed = ds_cropped.rename({lat_name: "latitude", lon_name: "longitude"})
        ds_regridded = ds_renamed.interp(
            latitude=target_lat,
            longitude=target_lon,
            method="linear"
        ).rename({"latitude": "lat", "longitude": "lon"})

    return ds_regridded


def align_time_coordinate(ds: xr.Dataset) -> xr.Dataset:
    """Ensure time coordinate is normalized to daily UTC midnight."""
    if "time" in ds.coords:
        daily_times = ds["time"].dt.floor("D")
        ds = ds.assign_coords(time=daily_times)
    return ds


def validate_regridded_shape(ds: xr.Dataset, label: str) -> None:
    """Abort if the regridded dataset does not match the 101x241 contract."""
    n_lat = len(ds["lat"]) if "lat" in ds.dims else len(ds["latitude"])
    n_lon = len(ds["lon"]) if "lon" in ds.dims else len(ds["longitude"])
    if n_lat != len(TARGET_LAT) or n_lon != len(TARGET_LON):
        raise ValueError(
            f"Shape mismatch in '{label}': expected ({len(TARGET_LAT)}, {len(TARGET_LON)}), got ({n_lat}, {n_lon})"
        )
    print(f"       Grid check ✅ {label}: lat={n_lat}, lon={n_lon}")


def harmonize_all_sources(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Collect, regrid, and merge all available surface and subsurface datasets.
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    surface_datasets = []

    print("=" * 65)
    print("  OceanEmbed — Spatial Harmonization & Multi-Source Regridding")
    print("=" * 65)
    print(f"  Target grid : {len(TARGET_LAT)} lat × {len(TARGET_LON)} lon (0.25° resolution)")
    print(f"  Lat range   : {TARGET_LAT[0]}°N to {TARGET_LAT[-1]}°N")
    print(f"  Lon range   : {TARGET_LON[0]}°E to {TARGET_LON[-1]}°E")
    print()

    # 1. SST: Prefer high-res processed SST (CMC satellite) or GLORYS surface
    sst_files = sorted(list((processed_dir / "sst").glob("*.nc")))
    if sst_files:
        print(f"  [1a] Loading high-resolution processed SST ({len(sst_files)} files)...")
        from src.utils.io import load_processed_variable
        ds_sst = load_processed_variable("sst", processed_dir=processed_dir)
        ds_sst = align_time_coordinate(ds_sst)
        if "sst" in ds_sst:
            surface_datasets.append(ds_sst[["sst"]])
            print("        ✅ Processed satellite SST loaded.")

    # 1b. GLORYS Surface Inputs (SSH, U/V currents, and fallback SST)
    glorys_surface_files = sorted(list(raw_dir.glob("*surface*.nc")))
    if glorys_surface_files:
        print(f"  [1b] Processing GLORYS surface inputs from {glorys_surface_files[0].name}...")
        ds_surf = xr.open_dataset(glorys_surface_files[0])
        if "depth" in ds_surf.dims:
            ds_surf = ds_surf.squeeze("depth", drop=True)
        ds_surf_reg = regrid_to_target(ds_surf)
        ds_surf_reg = align_time_coordinate(ds_surf_reg)

        var_rename = {}
        if "thetao" in ds_surf_reg and not any("sst" in d.data_vars for d in surface_datasets):
            var_rename["thetao"] = "sst"
        if "zos" in ds_surf_reg:
            var_rename["zos"] = "ssh"
        if "uo" in ds_surf_reg:
            var_rename["uo"] = "u_curr"
        if "vo" in ds_surf_reg:
            var_rename["vo"] = "v_curr"

        if var_rename:
            ds_surf_reg = ds_surf_reg.rename(var_rename)
        
        # Only keep standardized variables
        keep_vars = [v for v in ["sst", "ssh", "u_curr", "v_curr"] if v in ds_surf_reg]
        if keep_vars:
            surface_datasets.append(ds_surf_reg[keep_vars])
            print(f"        ✅ Regridded GLORYS variables: {keep_vars}")

    # 2. NASA SMAP SSS
    sss_files = sorted(list(raw_dir.glob("*sss*.nc")))
    if sss_files:
        print(f"  [2] Processing NASA SMAP SSS from {sss_files[0].name}...")
        ds_sss = xr.open_dataset(sss_files[0])
        ds_sss_reg = regrid_to_target(ds_sss)
        ds_sss_reg = align_time_coordinate(ds_sss_reg)

        if "sss_smap" in ds_sss_reg:
            ds_sss_reg = ds_sss_reg.rename({"sss_smap": "sss"})
        if "sss" in ds_sss_reg:
            surface_datasets.append(ds_sss_reg[["sss"]])
            print("        ✅ Regridded SSS variable.")

    # 3. ERA5 Winds
    wind_files = sorted(list(raw_dir.glob("*wind*.nc")))
    if wind_files:
        print(f"  [3] Processing ERA5 10m surface winds from {wind_files[0].name}...")
        ds_wind = xr.open_dataset(wind_files[0])
        ds_wind_reg = regrid_to_target(ds_wind)
        ds_wind_reg = align_time_coordinate(ds_wind_reg)

        var_rename = {}
        if "u10" in ds_wind_reg: var_rename["u10"] = "u_wind"
        if "v10" in ds_wind_reg: var_rename["v10"] = "v_wind"
        if var_rename:
            ds_wind_reg = ds_wind_reg.rename(var_rename)
        
        keep_vars = [v for v in ["u_wind", "v_wind"] if v in ds_wind_reg]
        if keep_vars:
            surface_datasets.append(ds_wind_reg[keep_vars])
            print(f"        ✅ Regridded ERA5 variables: {keep_vars}")

    out_surf_path = None
    if surface_datasets:
        print("\n  Merging surface variables into single dataset...")
        ds_surface_merged = xr.merge(surface_datasets, compat="override", join="inner")
        validate_regridded_shape(ds_surface_merged, "surface_inputs_merged")

        out_surf_path = processed_dir / "surface_inputs_regridded.nc"
        if "time" in ds_surface_merged.coords:
            ds_surface_merged["time"].encoding.clear()
        ds_surface_merged.to_netcdf(out_surf_path)
        print(f"  ✅ Saved unified surface file: {out_surf_path}")
        print(f"     Variables : {list(ds_surface_merged.data_vars)}")
        print(f"     Dimensions: {dict(ds_surface_merged.sizes)}")
    else:
        print("  ⚠️  No surface datasets available to merge yet.")

    # 4. GLORYS 3D Target Temperature
    out_tgt_path = None
    target_files = sorted(list(raw_dir.glob("*target*.nc")) + list(raw_dir.glob("*glorys_target*.nc")))
    if target_files:
        print(f"\n  [4] Processing GLORYS 3D Target Temperature from {target_files[0].name}...")
        ds_tgt = xr.open_dataset(target_files[0])
        ds_tgt_reg = regrid_to_target(ds_tgt)
        ds_tgt_reg = align_time_coordinate(ds_tgt_reg)

        depth_dim = "depth" if "depth" in ds_tgt_reg.dims else "deptht"
        ds_tgt_depths = ds_tgt_reg.sel({depth_dim: TARGET_DEPTHS}, method="nearest")

        validate_regridded_shape(ds_tgt_depths, "target_temp_regridded")
        n_depths = len(ds_tgt_depths[depth_dim])
        if n_depths != len(TARGET_DEPTHS):
            raise ValueError(f"Expected {len(TARGET_DEPTHS)} depth levels, got {n_depths}")

        out_tgt_path = processed_dir / "target_temp_regridded.nc"
        if "time" in ds_tgt_depths.coords:
            ds_tgt_depths["time"].encoding.clear()
        ds_tgt_depths.to_netcdf(out_tgt_path)
        print(f"  ✅ Saved target file: {out_tgt_path}")
        print(f"     Depth levels: {n_depths} levels extracted ({TARGET_DEPTHS})")
        print(f"     Dimensions  : {dict(ds_tgt_depths.sizes)}")

    print("\n" + "=" * 65)
    print("  ✅ Harmonization & Regridding complete!")
    print("=" * 65)
    return out_surf_path, out_tgt_path


if __name__ == "__main__":
    harmonize_all_sources()
