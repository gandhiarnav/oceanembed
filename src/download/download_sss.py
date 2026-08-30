"""
=============================================================================
OceanEmbed — Step 3b: Download Sea Surface Salinity (SSS) — SMAP L4 (NASA)
=============================================================================
Dataset   : SMAP_RSS_L4_SSS_8DAY_RUNNINGMEAN_V6  (RSS SMAP L4 SSS)
Variable  : sss_smap  — Sea Surface Salinity (PSU)
Source    : NASA PO.DAAC via earthaccess

Why SMAP L4 RSS?
  - Native resolution: 0.25° × 0.25°  ← already matches our target grid!
  - Global daily coverage (8-day running mean, updated every day)
  - Best spatial coverage in the Indian Ocean vs raw L2 swath data
  - No regridding needed — just crop to NIO bounding box

Resolution Note:
  SMAP L4 is an "8-day running mean" product, but a new file is produced
  every day. We treat each file as one daily snapshot. This is standard
  practice in oceanography (smoothed salinity changes slowly enough that
  this is valid for ML training input).

Region    : North Indian Ocean
  Latitude  : 5°N  to 30°N
  Longitude : 45°E to 105°E

Period    : 2020-01-01 to 2020-01-31  (1-month test sample)

Output    : ./data/raw/sss/  (one .nc file per day)
            ./data/raw/sss_merged_jan2020.nc  (merged single file)

Prerequisites:
  1. Create a FREE NASA Earthdata account: https://urs.earthdata.nasa.gov
  2. pip install earthaccess xarray netCDF4 numpy
  3. Run once:  earthaccess.login(strategy="interactive")
     OR set env vars:
       EARTHDATA_USERNAME=your_username
       EARTHDATA_PASSWORD=your_password
=============================================================================
"""

import os
import json
import numpy as np
import xarray as xr
import earthaccess
from dotenv import load_dotenv

# ── Load credentials from .env file ──────────────────────────────────────────
# Reads EARTHDATA_USERNAME and EARTHDATA_PASSWORD from .env in project root
load_dotenv()

# ── Output directories ────────────────────────────────────────────────────────
RAW_DIR    = "./data/raw/sss"       # one file per day goes here
OUTPUT_DIR = "./data/raw"            # merged output goes here
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Bounding Box: North Indian Ocean (must match all other scripts exactly) ───
MIN_LON = 45.0
MAX_LON = 105.0
MIN_LAT = 5.0
MAX_LAT = 30.0

# ── Time window: keep in sync with download_glorys.py ─────────────────────────
START_DATE = "2020-01-01"
END_DATE   = "2020-01-31"

# ── SMAP L4 Product Short Name on NASA Earthdata ──────────────────────────────
# Options (all at 0.25°, RSS product):
#   V6 (latest stable):   SMAP_RSS_L4_SSS_8DAY_RUNNINGMEAN_V6
#   V5 (fallback):        SMAP_RSS_L4_SSS_8DAY_RUNNINGMEAN_V5
SMAP_SHORT_NAME = "SMAP_RSS_L4_SSS_8DAY_RUNNINGMEAN_V6"

# ── Target variable name inside the NetCDF file ───────────────────────────────
# Inspect with: ds.data_vars  if unsure
SMAP_VAR = "sss_smap"   # primary SSS field in RSS L4 product

print("=" * 65)
print("  Downloading SMAP L4 Sea Surface Salinity (RSS, 0.25°)")
print("=" * 65)
print(f"  Product   : {SMAP_SHORT_NAME}")
print(f"  Variable  : {SMAP_VAR} (PSU)")
print(f"  Region    : {MIN_LAT}°N–{MAX_LAT}°N, {MIN_LON}°E–{MAX_LON}°E")
print(f"  Period    : {START_DATE}  →  {END_DATE}")
print(f"  Raw files : {RAW_DIR}/")
print(f"  Merged    : {OUTPUT_DIR}/sss_merged_jan2020.nc")
print()

# ── Step 1: Authenticate with NASA Earthdata ──────────────────────────────────
print("  [1/4] Authenticating with NASA Earthdata...")
print("        (Uses EARTHDATA_USERNAME / EARTHDATA_PASSWORD env vars,")
print("         or will prompt interactively if not set)")
print()

auth = earthaccess.login(strategy="environment")   # change to "interactive" if env vars not set
print(f"  ✅ Logged in as: {auth.username}")
print()

# ── Step 2: Search for granules in the time window ────────────────────────────
print(f"  [2/4] Searching for SMAP granules ({START_DATE} → {END_DATE})...")

results = earthaccess.search_data(
    short_name   = SMAP_SHORT_NAME,
    temporal     = (START_DATE, END_DATE),
    bounding_box = (MIN_LON, MIN_LAT, MAX_LON, MAX_LAT),   # (W, S, E, N)
)

print(f"  ✅ Found {len(results)} granule(s)")
print()

if len(results) == 0:
    print("  ⚠️  No granules found. Check date range or product short name.")
    print("     Try: https://search.earthdata.nasa.gov/  to browse SMAP products.")
    raise SystemExit(1)

# ── Step 3: Download raw files ────────────────────────────────────────────────
print(f"  [3/4] Downloading {len(results)} granule(s) → {RAW_DIR}/")
print()

downloaded_files = earthaccess.download(results, local_path=RAW_DIR)

print()
print(f"  ✅ Downloaded {len(downloaded_files)} file(s)")
print()

# ── Step 4: Crop to NIO bounding box & merge into one file ───────────────────
print(f"  [4/4] Cropping to NIO box and merging into single NetCDF...")

datasets = []
for fpath in sorted(downloaded_files):
    try:
        ds = xr.open_dataset(fpath)

        # ── Normalise coordinate names ─────────────────────────────────────
        # SMAP RSS files use 'lat' and 'lon' (confirm with: print(ds.coords))
        lat_name = "lat"  if "lat"  in ds.coords else "latitude"
        lon_name = "lon"  if "lon"  in ds.coords else "longitude"

        # ── Crop to NIO bounding box ───────────────────────────────────────
        ds_crop = ds.sel(
            {lat_name: slice(MIN_LAT, MAX_LAT),
             lon_name: slice(MIN_LON, MAX_LON)}
        )

        # ── Keep only the SSS variable + rename for consistency ────────────
        if SMAP_VAR in ds_crop.data_vars:
            ds_crop = ds_crop[[SMAP_VAR]].rename({SMAP_VAR: "sss"})
        else:
            # Fallback: print what variables are available and pick the first
            available = list(ds_crop.data_vars)
            print(f"  ⚠️  '{SMAP_VAR}' not found. Available: {available}")
            print(f"      Using '{available[0]}' — update SMAP_VAR if wrong.")
            ds_crop = ds_crop[[available[0]]].rename({available[0]: "sss"})

        datasets.append(ds_crop)
        ds.close()

    except Exception as e:
        print(f"  ⚠️  Could not open {fpath}: {e}")

# ── Merge all daily slices along the time dimension ───────────────────────────
ds_merged = xr.concat(datasets, dim="time")
ds_merged = ds_merged.sortby("time")

merged_path = os.path.join(OUTPUT_DIR, "sss_merged_jan2020.nc")
ds_merged.to_netcdf(merged_path)

print(f"  ✅ Merged file written: {merged_path}")
print()

# ── Step 5: Quick sanity check ────────────────────────────────────────────────
print("  ─── Quick Sanity Check ───────────────────────────────────────")

sss_vals = ds_merged["sss"].values
ocean_vals = sss_vals[~np.isnan(sss_vals)]

n_times = ds_merged.dims.get("time", "?")
lat_arr  = ds_merged.coords.get("lat",  ds_merged.coords.get("latitude")).values
lon_arr  = ds_merged.coords.get("lon",  ds_merged.coords.get("longitude")).values
lat_res  = float(np.diff(lat_arr).mean())
lon_res  = float(np.diff(lon_arr).mean())
nan_frac = float(np.isnan(sss_vals).mean()) * 100

print(f"  ✅ Time steps    : {n_times}  (expected ~31 for January)")
print(f"  ✅ Grid shape    : {sss_vals.shape[1]} lat × {sss_vals.shape[2]} lon")
print(f"  ✅ Resolution    : lat {lat_res:.4f}°, lon {lon_res:.4f}°  (target: ~0.25°)")
print(f"  ✅ Salinity range: {ocean_vals.min():.2f} – {ocean_vals.max():.2f} PSU")
print(f"  ✅ NaN fraction  : {nan_frac:.1f}%  (land + gaps — expected)")

if ocean_vals.min() < 20 or ocean_vals.max() > 42:
    print("  ⚠️  WARNING: Salinity outside 20–42 PSU range. Inspect for outliers.")

needs_regrid = not (0.20 < lat_res < 0.30 and 0.20 < lon_res < 0.30)
if needs_regrid:
    print()
    print("  ⚠️  Resolution is NOT 0.25°. Regridding will be needed in regrid_harmonize.py.")
    print("      Update the SSS section there to use bilinear interpolation.")
else:
    print()
    print("  ✅ Resolution is already 0.25° — no regridding needed in regrid_harmonize.py!")
    print("     Just crop + rename coordinates in the harmonization step.")

ds_merged.close()

print()
print("  ─── SSS Download COMPLETE ────────────────────────────────────")
print(f"  File ready : {merged_path}")
print()

# ── Save metadata for pipeline tracking ───────────────────────────────────────
meta = {
    "source"      : "NASA PO.DAAC — SMAP L4 RSS",
    "product"     : SMAP_SHORT_NAME,
    "variable_raw": SMAP_VAR,
    "variable_out": "sss",
    "units"       : "PSU",
    "resolution"  : "0.25°",
    "time_period" : f"{START_DATE} to {END_DATE}",
    "n_timesteps" : int(n_times) if isinstance(n_times, int) else n_times,
    "bounding_box": {"min_lat": MIN_LAT, "max_lat": MAX_LAT,
                     "min_lon": MIN_LON, "max_lon": MAX_LON},
    "output_file" : merged_path,
    "needs_regrid": needs_regrid,
}
meta_path = os.path.join(OUTPUT_DIR, "sss_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"  Metadata saved: {meta_path}")
print()
print("  Next step: Run regrid_harmonize.py to align SSS with SST, SSH, Winds.")
