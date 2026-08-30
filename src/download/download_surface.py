"""
=============================================================================
OceanEmbed — Step 5: Download GLORYS12V1 Surface Input Variables
=============================================================================
Dataset   : cmems_mod_glo_phy_my_0.083deg_P1D-m  (GLORYS12V1 Daily)
Variables : thetao (SST), zos (SSH), uo (U-current), vo (V-current)

Region    : North Indian Ocean
  Latitude  : 5°N  to 30°N
  Longitude : 45°E to 105°E

Depth     : 0 to 0.5 m  (SURFACE LAYER ONLY — first GLORYS level ~0.49m)
Period    : 2020-01-01 to 2020-01-31  (1-month test sample)

Output    : ./data/raw/glorys_surface_inputs_jan2020.nc

Variable Mapping:
  thetao → SST  (Sea Surface Temperature, °C)
  zos    → SSH  (Sea Surface Height / Sea Level Anomaly, m)
  uo     → U_c  (Surface Current eastward component, m/s)
  vo     → V_c  (Surface Current northward component, m/s)

Note: SSS and surface winds (ERA5) are downloaded by separate scripts.
      This file provides 4 of the 7 required input channels.
=============================================================================
"""

import copernicusmarine
import os

# ── Output directory ──────────────────────────────────────────────────────────
OUTPUT_DIR = "./data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Bounding Box: North Indian Ocean ─────────────────────────────────────────
MIN_LON = 45.0
MAX_LON = 105.0
MIN_LAT = 5.0
MAX_LAT = 30.0

# ── Depth: surface layer ONLY ─────────────────────────────────────────────────
# GLORYS's first depth level is ~0.49m — restricting to 0–0.5m isolates it
MIN_DEPTH = 0.0
MAX_DEPTH = 0.5      # surface layer only (first model level, ~0.49 m)

# ── Time window ───────────────────────────────────────────────────────────────
START_DATE = "2020-01-01T00:00:00"
END_DATE   = "2020-01-31T23:59:59"

print("=" * 65)
print("  Downloading GLORYS12V1 — Surface Input Variables")
print("=" * 65)
print(f"  Variables : thetao (SST), zos (SSH), uo (U-curr), vo (V-curr)")
print(f"  Region    : {MIN_LAT}°N–{MAX_LAT}°N, {MIN_LON}°E–{MAX_LON}°E")
print(f"  Depth     : {MIN_DEPTH} – {MAX_DEPTH} m  (surface only)")
print(f"  Period    : {START_DATE}  →  {END_DATE}")
print(f"  Output    : {OUTPUT_DIR}/glorys_surface_inputs_jan2020.nc")
print()

copernicusmarine.subset(
    dataset_id        = "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables         = ["thetao", "zos", "uo", "vo"],
    start_datetime    = START_DATE,
    end_datetime      = END_DATE,
    minimum_longitude = MIN_LON,
    maximum_longitude = MAX_LON,
    minimum_latitude  = MIN_LAT,
    maximum_latitude  = MAX_LAT,
    minimum_depth     = MIN_DEPTH,
    maximum_depth     = MAX_DEPTH,
    output_filename   = "glorys_surface_inputs_jan2020.nc",
    output_directory  = OUTPUT_DIR,
)

print()
print("  ✅ Surface inputs download complete!")
print(f"     File: {OUTPUT_DIR}/glorys_surface_inputs_jan2020.nc")
print()
print("  You now have:")
print("    glorys_target_temp_jan2020.nc  → 3D temperature for training")
print("    glorys_surface_inputs_jan2020.nc → SST, SSH, U/V currents")
print()
print("  Next step: Run regrid_harmonize.py to standardize to 0.25° grid.")
