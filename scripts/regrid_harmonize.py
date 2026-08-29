"""
=============================================================================
OceanEmbed — Day 3: Spatial Harmonization & Regridding
=============================================================================
Purpose : Force all downloaded datasets onto the exact same
          0.25° × 0.25° grid covering the North Indian Ocean.

Target Grid:
  Latitude  : 5.0°N  to 30.0°N  at 0.25° → 101 points
  Longitude : 45.0°E to 105.0°E at 0.25° → 241 points

Input  : data/raw/*.nc        (GLORYS files at ~1/12° native resolution)
Output : data/processed/*.nc  (All variables regridded to 0.25°)

Method : Bilinear interpolation using xESMF
         (Conservative regridding available if mass conservation needed)
=============================================================================
"""

import xarray as xr
import numpy as np
import os

try:
    import xesmf as xe
    XESMF_AVAILABLE = True
except ImportError:
    XESMF_AVAILABLE = False
    print("  ⚠️  xESMF not installed. Falling back to xarray interp().")
    print("     Install with: pip install xesmf")

# ── Directory setup ───────────────────────────────────────────────────────────
RAW_DIR       = "./data/raw"
PROCESSED_DIR = "./data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Master Target Grid Definition ─────────────────────────────────────────────
LAT_TARGET = np.arange(5.0, 30.25, 0.25)    # 101 points
LON_TARGET = np.arange(45.0, 105.25, 0.25)  # 241 points

print("=" * 65)
print("  OceanEmbed — Spatial Harmonization & Regridding")
print("=" * 65)
print(f"  Target grid: {len(LAT_TARGET)} lat × {len(LON_TARGET)} lon")
print(f"  Lat: {LAT_TARGET[0]}°N to {LAT_TARGET[-1]}°N")
print(f"  Lon: {LON_TARGET[0]}°E to {LON_TARGET[-1]}°E")
print()


def regrid_to_target(ds: xr.Dataset, lat_name: str = "latitude",
                     lon_name: str = "longitude") -> xr.Dataset:
    """
    Regrid an xarray Dataset to the target 0.25° NIO grid.

    Parameters
    ----------
    ds       : Input dataset (any resolution)
    lat_name : Name of the latitude dimension in ds
    lon_name : Name of the longitude dimension in ds

    Returns
    -------
    ds_regridded : Dataset on the 101×241 target grid
    """
    # Step 1: Crop to NIO bounding box first (faster processing)
    ds = ds.sel(
        {lat_name: slice(4.0, 31.0),   # slight buffer for interpolation edges
         lon_name: slice(44.0, 106.0)}
    )

    if XESMF_AVAILABLE:
        # Build target grid dataset
        ds_out = xr.Dataset({
            "lat": (["lat"], LAT_TARGET),
            "lon": (["lon"], LON_TARGET),
        })
        # Rename source dims to match xESMF conventions
        ds = ds.rename({lat_name: "lat", lon_name: "lon"})
        regridder = xe.Regridder(ds, ds_out, "bilinear",
                                 periodic=False, ignore_degenerate=True)
        ds_regridded = regridder(ds, keep_attrs=True)
    else:
        # Fallback: xarray linear interpolation
        ds = ds.rename({lat_name: "latitude", lon_name: "longitude"})
        ds_regridded = ds.interp(
            latitude=LAT_TARGET,
            longitude=LON_TARGET,
            method="linear"
        ).rename({"latitude": "lat", "longitude": "lon"})

    return ds_regridded


def verify_grid(ds: xr.Dataset) -> None:
    """Print grid verification summary."""
    lat_ok = len(ds.lat) == 101
    lon_ok = len(ds.lon) == 241
    print(f"    Lat points : {len(ds.lat)} {'✅' if lat_ok else '❌ (expected 101)'}")
    print(f"    Lon points : {len(ds.lon)} {'✅' if lon_ok else '❌ (expected 241)'}")
    print(f"    Variables  : {list(ds.data_vars)}")
    nan_pct = float(np.isnan(ds[list(ds.data_vars)[0]].values).mean() * 100)
    print(f"    NaN pixels : {nan_pct:.1f}% (land + missing data)")


# ── Process GLORYS Surface Inputs ─────────────────────────────────────────────
surface_file = os.path.join(RAW_DIR, "glorys_surface_inputs_jan2020.nc")
if os.path.exists(surface_file):
    print("  Processing: glorys_surface_inputs_jan2020.nc")
    ds_surf = xr.open_dataset(surface_file)
    print(f"    Native grid: {ds_surf.dims}")

    # GLORYS uses 'latitude'/'longitude' — adjust if your file differs
    lat_dim = "latitude" if "latitude" in ds_surf.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds_surf.dims else "lon"

    # Squeeze out depth dimension (surface variables have depth=1)
    if "depth" in ds_surf.dims:
        ds_surf = ds_surf.squeeze("depth", drop=True)

    ds_surf_regridded = regrid_to_target(ds_surf, lat_name=lat_dim, lon_name=lon_dim)
    out_path = os.path.join(PROCESSED_DIR, "surface_inputs_regridded.nc")
    ds_surf_regridded.to_netcdf(out_path)
    print(f"    Saved: {out_path}")
    verify_grid(ds_surf_regridded)
    print()
else:
    print(f"  ⚠️  Surface file not found: {surface_file}")
    print("     Run scripts/download_surface.py first.")
    print()

# ── Process GLORYS 3D Target ──────────────────────────────────────────────────
target_file = os.path.join(RAW_DIR, "glorys_target_temp_jan2020.nc")
STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

if os.path.exists(target_file):
    print("  Processing: glorys_target_temp_jan2020.nc (3D temperature)")
    ds_tgt = xr.open_dataset(target_file)
    print(f"    Native grid: {ds_tgt.dims}")

    lat_dim = "latitude" if "latitude" in ds_tgt.dims else "lat"
    lon_dim = "longitude" if "longitude" in ds_tgt.dims else "lon"

    # Regrid the horizontal dimensions
    ds_tgt_regridded = regrid_to_target(ds_tgt, lat_name=lat_dim, lon_name=lon_dim)

    # Extract the 15 standard depth levels
    depth_dim = "depth" if "depth" in ds_tgt_regridded.dims else "deptht"
    ds_tgt_depths = ds_tgt_regridded.sel(
        {depth_dim: STANDARD_DEPTHS}, method="nearest"
    )

    out_path = os.path.join(PROCESSED_DIR, "target_temp_regridded.nc")
    ds_tgt_depths.to_netcdf(out_path)
    print(f"    Saved: {out_path}")
    verify_grid(ds_tgt_depths)
    print(f"    Depth levels: {len(ds_tgt_depths[depth_dim])} levels extracted")
    print()
else:
    print(f"  ⚠️  Target file not found: {target_file}")
    print("     Run scripts/download_glorys.py first.")
    print()

print("  ✅ Regridding complete!")
print("     Next step: Run normalize.py to z-score normalize the inputs.")
