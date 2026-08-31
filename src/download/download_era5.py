"""
OceanEmbed — Download ERA5 Surface Winds (10m U/V Wind Components)
Dataset: reanalysis-era5-single-levels
Variables: 10m_u_component_of_wind (u10), 10m_v_component_of_wind (v10)
Source: ECMWF Copernicus Climate Data Store (CDS)
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import cdsapi
import xarray as xr
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, RAW_DATA_DIR

load_dotenv()


def download_era5_winds(
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-31",
    output_filename: str = "era5_winds_jan2020.nc",
    output_dir: Path = RAW_DATA_DIR
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / output_filename
    tmp_hourly_file = output_dir / f"tmp_hourly_{output_filename}"

    # Parse date range
    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")
    delta = (dt_end - dt_start).days + 1
    dates = [dt_start + timedelta(days=i) for i in range(delta)]

    years = sorted(list({f"{d.year}" for d in dates}))
    months = sorted(list({f"{d.month:02d}" for d in dates}))
    days = sorted(list({f"{d.day:02d}" for d in dates}))
    hours = [f"{h:02d}:00" for h in range(0, 24, 6)]

    print("=" * 65)
    print("  Downloading ERA5 Surface Winds (10m U/V Wind Components)")
    print("=" * 65)
    print(f"  Variables : 10m_u_component_of_wind, 10m_v_component_of_wind")
    print(f"  Region    : {LAT_MIN}°N–{LAT_MAX}°N, {LON_MIN}°E–{LON_MAX}°E")
    print(f"  Period    : {start_date} → {end_date}")
    print(f"  Output    : {out_file}")
    print()

    cds_url = os.getenv("CDSAPI_URL")
    cds_key = os.getenv("CDSAPI_KEY")

    if cds_url and cds_key and not cds_key.startswith("your-"):
        client = cdsapi.Client(url=cds_url, key=cds_key)
    else:
        try:
            client = cdsapi.Client()
        except Exception as e:
            print("❌ CDS API credentials not configured.")
            print("   Please set CDSAPI_URL and CDSAPI_KEY in .env")
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
                "year": years,
                "month": months,
                "day": days,
                "time": hours,
                "area": [LAT_MAX, LON_MIN, LAT_MIN, LON_MAX],  # [N, W, S, E]
            },
            str(tmp_hourly_file)
        )
        print("  ✅ Download completed.")
    except Exception as e:
        print(f"  ❌ Error downloading ERA5 wind data: {e}")
        sys.exit(1)

    print("  [2/3] Resampling to daily averages...")
    try:
        ds = xr.open_dataset(tmp_hourly_file)

        time_name = "valid_time" if "valid_time" in ds.coords else ("time" if "time" in ds.coords else None)
        if time_name != "time" and time_name is not None:
            ds = ds.rename({time_name: "time"})

        ds_daily = ds.resample(time="1D").mean()

        rename_dict = {}
        if "u10" not in ds_daily.data_vars:
            for v in ds_daily.data_vars:
                if "u" in v.lower():
                    rename_dict[v] = "u10"
                elif "v" in v.lower():
                    rename_dict[v] = "v10"
        if rename_dict:
            ds_daily = ds_daily.rename(rename_dict)

        if "time" in ds_daily.coords:
            ds_daily["time"].encoding.clear()

        ds_daily.to_netcdf(out_file)
        ds.close()
        ds_daily.close()

        if tmp_hourly_file.exists():
            tmp_hourly_file.unlink()

        print(f"  ✅ Daily wind file saved: {out_file}")
    except Exception as e:
        print(f"  ❌ Error processing ERA5 netCDF: {e}")
        sys.exit(1)

    print("  [3/3] Done! Surface winds ready for harmonization.")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Download ERA5 surface winds.")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2020-01-31", help="End date YYYY-MM-DD")
    parser.add_argument("--output", type=str, default="era5_winds_jan2020.nc", help="Output filename")

    args = parser.parse_args()
    download_era5_winds(start_date=args.start, end_date=args.end, output_filename=args.output)


if __name__ == "__main__":
    main()
