"""
=============================================================================
OceanEmbed — Step 3c: Download ERA5 Surface Winds (U-wind, V-wind)
=============================================================================
Dataset   : reanalysis-era5-single-levels
Variables : 10m_u_component_of_wind (u10), 10m_v_component_of_wind (v10)
Source    : ECMWF Copernicus Climate Data Store (CDS)

Region    : North Indian Ocean
  Latitude  : 5°N  to 30°N
  Longitude : 45°E to 105°E

Period    : 2020-01-01 to 2020-01-31 (1-month test sample, daily aggregated)
Output    : ./data/raw/era5_winds_jan2020.nc

Prerequisites:
  1. Create free account: https://cds.climate.copernicus.eu
  2. Set CDSAPI_URL and CDSAPI_KEY in .env
     OR configure ~/.cdsapirc
  3. pip install cdsapi python-dotenv xarray
=============================================================================
"""

import os
import sys
import cdsapi
import xarray as xr
from dotenv import load_dotenv

# ── Load credentials from .env ───────────────────────────────────────────────
load_dotenv()

OUTPUT_DIR = "./data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "era5_winds_jan2020.nc")
TMP_HOURLY_FILE = os.path.join(OUTPUT_DIR, "era5_winds_hourly_raw.nc")

# ── Bounding Box: [North, West, South, East] for CDS API ─────────────────────
NORTH = 30.0
WEST  = 45.0
SOUTH = 5.0
EAST  = 105.0

START_YEAR  = "2020"
START_MONTH = "01"
DAYS = [f"{d:02d}" for d in range(1, 32)]
HOURS = [f"{h:02d}:00" for h in range(0, 24, 6)]  # sample every 6h for efficient daily mean

print("=" * 65)
print("  Downloading ERA5 Surface Winds (10m U/V Wind Components)")
print("=" * 65)
print(f"  Variables : 10m_u_component_of_wind, 10m_v_component_of_wind")
print(f"  Region    : {SOUTH}°N–{NORTH}°N, {WEST}°E–{EAST}°E")
print(f"  Period    : {START_YEAR}-{START_MONTH}-01 to {START_YEAR}-{START_MONTH}-31")
print(f"  Output    : {OUTPUT_FILE}")
print()

# Check CDS API configuration
cds_url = os.getenv("CDSAPI_URL")
cds_key = os.getenv("CDSAPI_KEY")

if cds_url and cds_key and not cds_key.startswith("your-"):
    client = cdsapi.Client(url=cds_url, key=cds_key)
else:
    # Try default ~/.cdsapirc
    try:
        client = cdsapi.Client()
    except Exception as e:
        print("  ❌ CDS API credentials not configured.")
        print("     Please set CDSAPI_URL and CDSAPI_KEY in your .env file.")
        print("     Get your key from: https://cds.climate.copernicus.eu/profile")
        sys.exit(1)

print("  [1/3] Submitting request to Copernicus Climate Data Store...")

try:
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
            ],
            "year": START_YEAR,
            "month": START_MONTH,
            "day": DAYS,
            "time": HOURS,
            "area": [NORTH, WEST, SOUTH, EAST],  # [N, W, S, E]
        },
        TMP_HOURLY_FILE
    )
    print("  ✅ Download completed.")
except Exception as e:
    print(f"  ❌ Error downloading ERA5 wind data: {e}")
    sys.exit(1)

print("  [2/3] Resampling to daily averages...")
try:
    ds = xr.open_dataset(TMP_HOURLY_FILE)
    
    # Coordinate name normalization
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    time_name = "valid_time" if "valid_time" in ds.coords else ("time" if "time" in ds.coords else None)
    
    if time_name != "time" and time_name is not None:
        ds = ds.rename({time_name: "time"})
    
    # Daily aggregation
    ds_daily = ds.resample(time="1D").mean()
    
    # Rename variables if needed (u10, v10)
    rename_dict = {}
    if "u10" not in ds_daily.data_vars:
        for v in ds_daily.data_vars:
            if "u" in v.lower():
                rename_dict[v] = "u10"
            elif "v" in v.lower():
                rename_dict[v] = "v10"
    if rename_dict:
        ds_daily = ds_daily.rename(rename_dict)
        
    ds_daily.to_netcdf(OUTPUT_FILE)
    ds.close()
    ds_daily.close()
    
    if os.path.exists(TMP_HOURLY_FILE):
        os.remove(TMP_HOURLY_FILE)
        
    print(f"  ✅ Daily wind file saved: {OUTPUT_FILE}")
except Exception as e:
    print(f"  ❌ Error processing ERA5 netCDF: {e}")
    sys.exit(1)

print("  [3/3] Done! Surface winds ready for harmonization.")
