"""
SST Preprocessing Pipeline for CMC 0.1° Global Level-4 SST.
Harmonizes raw NetCDF data to the SIH target grid (5-30°N, 45-105°E at 0.25° resolution).
"""
from pathlib import Path
from typing import Union, List, Optional
import numpy as np
import xarray as xr

from src.config import TARGET_LAT, TARGET_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX


def process_single_sst(
    input_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    target_lat: np.ndarray = TARGET_LAT,
    target_lon: np.ndarray = TARGET_LON,
) -> xr.Dataset:
    """
    Process a single daily CMC SST NetCDF file:
    1. Open dataset and select regional bounding box.
    2. Convert SST from Kelvin to Celsius.
    3. Regrid SST to 0.25° target grid (linear interpolation).
    4. Regrid land/sea mask to 0.25° target grid (nearest-neighbor).
    5. Mask land/non-ocean pixels.
    6. Return clean standardized xr.Dataset (and optionally save to disk).

    Parameters
    ----------
    input_file : str or Path
        Path to raw CMC SST NetCDF file.
    output_file : str or Path, optional
        Path to save the processed NetCDF file.
    target_lat : np.ndarray, optional
        Target latitude grid (default: 5.0 to 30.0 at 0.25°).
    target_lon : np.ndarray, optional
        Target longitude grid (default: 45.0 to 105.0 at 0.25°).

    Returns
    -------
    xr.Dataset
        Processed dataset containing 'sst' (and 'mask') on target grid.
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Open dataset
    with xr.open_dataset(input_path) as ds:
        # Buffer slightly for clean interpolation at the boundary
        lat_buf = 0.5
        lon_buf = 0.5
        
        # Crop region
        sst_raw = ds["analysed_sst"].sel(
            lat=slice(LAT_MIN - lat_buf, LAT_MAX + lat_buf),
            lon=slice(LON_MIN - lon_buf, LON_MAX + lon_buf)
        )
        mask_raw = ds["mask"].sel(
            lat=slice(LAT_MIN - lat_buf, LAT_MAX + lat_buf),
            lon=slice(LON_MIN - lon_buf, LON_MAX + lon_buf)
        )
        
        # Convert Kelvin to Celsius
        sst_celsius = sst_raw - 273.15
        
        # Regrid SST linearly to target grid
        sst_regridded = sst_celsius.interp(
            lat=target_lat,
            lon=target_lon,
            method="linear"
        )
        
        # Regrid mask using nearest neighbor
        mask_regridded = mask_raw.interp(
            lat=target_lat,
            lon=target_lon,
            method="nearest"
        )
        
        # Ocean mask: mask == 1 is water
        ocean_mask = (mask_regridded == 1)
        
        # Apply mask (land becomes NaN)
        sst_masked = sst_regridded.where(ocean_mask)
        
        # Build clean output dataset
        out_ds = xr.Dataset(
            data_vars={
                "sst": (
                    ["time", "lat", "lon"],
                    sst_masked.values,
                    {
                        "units": "degree_Celsius",
                        "long_name": "Sea Surface Temperature",
                        "standard_name": "sea_surface_temperature"
                    }
                ),
                "ocean_mask": (
                    ["time", "lat", "lon"],
                    ocean_mask.values.astype(np.int8),
                    {
                        "long_name": "Ocean Land Mask (1=ocean, 0=land/other)",
                        "flag_values": [0, 1],
                        "flag_meanings": "land ocean"
                    }
                )
            },
            coords={
                "time": ds["time"].values,
                "lat": target_lat,
                "lon": target_lon
            },
            attrs={
                "title": "Harmonized Daily SST (North Indian Ocean)",
                "source_file": input_path.name,
                "spatial_resolution": "0.25 degree",
                "domain": f"Lat: [{LAT_MIN}, {LAT_MAX}], Lon: [{LON_MIN}, {LON_MAX}]"
            }
        )

    if output_file is not None:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_ds.to_netcdf(out_path)

    return out_ds


def process_sst_batch(
    file_paths: List[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    save_individual: bool = True
) -> List[Path]:
    """
    Process a list of raw CMC SST files.

    Parameters
    ----------
    file_paths : list of (str or Path)
        List of paths to raw files.
    output_dir : str or Path, optional
        Directory where processed NetCDF files will be stored.
    save_individual : bool
        Whether to save individual daily processed NetCDF files.

    Returns
    -------
    list of Path
        Paths to saved processed files.
    """
    processed_paths = []
    out_dir_path = Path(output_dir) if output_dir else None

    if out_dir_path:
        out_dir_path.mkdir(parents=True, exist_ok=True)

    for fp in sorted(file_paths):
        path = Path(fp)
        # Extract date from filename if possible (CMC format: YYYYMMDD...)
        # or load dataset time
        ds = xr.open_dataset(path)
        date_str = str(np.datetime_as_string(ds.time.values[0], unit="D"))
        ds.close()

        out_file = None
        if save_individual and out_dir_path:
            out_file = out_dir_path / f"sst_{date_str}_0.25deg.nc"

        print(f"Processing SST for {date_str} from {path.name}...")
        process_single_sst(path, output_file=out_file)
        
        if out_file:
            processed_paths.append(out_file)

    return processed_paths
