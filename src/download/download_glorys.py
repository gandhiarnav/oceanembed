"""
OceanEmbed — Download GLORYS12V1 3D Temperature (Training Target)
Dataset: cmems_mod_glo_phy_my_0.083deg_P1D-m (GLORYS12V1 Daily)
Variable: thetao — Potential Temperature (°C)
Depth: 0 to 1000 m
"""
import os
import sys
import argparse
from pathlib import Path
import copernicusmarine
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RAW_DATA_DIR

load_dotenv()


def download_glorys_target(
    start_date: str = "2020-01-01T00:00:00",
    end_date: str = "2020-01-31T23:59:59",
    output_filename: str = "glorys_target_temp_jan2020.nc",
    output_dir: Path = RAW_DATA_DIR
):
    user = os.getenv("COPERNICUSMARINE_SERVICE_USERNAME") or os.getenv("COPERNICUS_USERNAME")
    pwd = os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD") or os.getenv("COPERNICUS_PASSWORD")

    if not user or not pwd:
        print("❌ Copernicus Marine credentials not found in .env.")
        print("   Please set COPERNICUSMARINE_SERVICE_USERNAME and COPERNICUSMARINE_SERVICE_PASSWORD.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  Downloading GLORYS12V1 — 3D Temperature (Training Target)")
    print("=" * 65)
    print(f"  Variable  : thetao (Potential Temperature)")
    print(f"  Region    : {LAT_MIN}°N–{LAT_MAX}°N, {LON_MIN}°E–{LON_MAX}°E")
    print(f"  Depth     : 0.0 – 1070.0 m (covers 1000m interpolation bound)")
    print(f"  Period    : {start_date} → {end_date}")
    print(f"  Output    : {output_dir / output_filename}")
    print()

    copernicusmarine.subset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao"],
        start_datetime=start_date,
        end_datetime=end_date,
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        minimum_depth=0.0,
        maximum_depth=1070.0,
        output_filename=output_filename,
        output_directory=str(output_dir),
        username=user,
        password=pwd,
        overwrite=True,
    )

    print()
    print("  ✅ 3D temperature download complete!")
    print(f"     File: {output_dir / output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Download GLORYS 3D target temperature.")
    parser.add_argument("--start", type=str, default="2020-01-01T00:00:00", help="Start datetime ISO")
    parser.add_argument("--end", type=str, default="2020-01-31T23:59:59", help="End datetime ISO")
    parser.add_argument("--output", type=str, default="glorys_target_temp_jan2020.nc", help="Output filename")

    args = parser.parse_args()
    download_glorys_target(start_date=args.start, end_date=args.end, output_filename=args.output)


if __name__ == "__main__":
    main()
