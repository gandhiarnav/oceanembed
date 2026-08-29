"""
=============================================================================
OceanEmbed — Step 4: Download GLORYS12V1 3D Temperature (Training Target)
=============================================================================
Dataset   : cmems_mod_glo_phy_my_0.083deg_P1D-m  (GLORYS12V1 Daily)
Variable  : thetao  — Potential Temperature (°C)

Region    : North Indian Ocean
  Latitude  : 5°N  to 30°N
  Longitude : 45°E to 105°E

Depth     : 0 to 1000 m  (full 3D profile — ALL 15 target depth levels)
Period    : 2020-01-01 to 2020-01-31  (1-month test sample)

Output    : ./data/raw/glorys_target_temp_jan2020.nc

Prerequisites:
  pip install copernicusmarine
  copernicusmarine login   ← run once to save credentials
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

# ── Depth: full profile covering all 15 target levels ────────────────────────
# GLORYS standard depths include: 0.49, 1.54, 2.65, 4.59, 7.26, 11.40,
# 17.44, 25.21, 36.89 ... up to 5902m. We subset 0–1000m.
MIN_DEPTH = 0.0
MAX_DEPTH = 1000.0

# ── Time window (1-month test sample) ────────────────────────────────────────
START_DATE = "2020-01-01T00:00:00"
END_DATE   = "2020-01-31T23:59:59"

print("=" * 65)
print("  Downloading GLORYS12V1 — 3D Temperature (Training Target)")
print("=" * 65)
print(f"  Variable  : thetao (Potential Temperature)")
print(f"  Region    : {MIN_LAT}°N–{MAX_LAT}°N, {MIN_LON}°E–{MAX_LON}°E")
print(f"  Depth     : {MIN_DEPTH} – {MAX_DEPTH} m")
print(f"  Period    : {START_DATE}  →  {END_DATE}")
print(f"  Output    : {OUTPUT_DIR}/glorys_target_temp_jan2020.nc")
print()

copernicusmarine.subset(
    dataset_id        = "cmems_mod_glo_phy_my_0.083deg_P1D-m",
    variables         = ["thetao"],
    start_datetime    = START_DATE,
    end_datetime      = END_DATE,
    minimum_longitude = MIN_LON,
    maximum_longitude = MAX_LON,
    minimum_latitude  = MIN_LAT,
    maximum_latitude  = MAX_LAT,
    minimum_depth     = MIN_DEPTH,
    maximum_depth     = MAX_DEPTH,
    output_filename   = "glorys_target_temp_jan2020.nc",
    output_directory  = OUTPUT_DIR,
)

print()
print("  ✅ 3D temperature download complete!")
print(f"     File: {OUTPUT_DIR}/glorys_target_temp_jan2020.nc")
print()
print("  Next step: Run download_surface.py to get surface input variables.")
