"""
Utility functions for reading and writing processed SIH ocean datasets.
"""
from pathlib import Path
from typing import Optional, Union, List
import xarray as xr

from src.config import PROCESSED_DATA_DIR


def load_processed_variable(
    var_name: str = "sst",
    processed_dir: Optional[Union[str, Path]] = None,
    sort_by_time: bool = True
) -> xr.Dataset:
    """
    Load and concatenate all processed daily NetCDF files for a given variable.

    Parameters
    ----------
    var_name : str
        Variable name directory (e.g., 'sst', 'sss', 'ssh').
    processed_dir : str or Path, optional
        Base processed directory (default: data/processed).
    sort_by_time : bool
        Whether to sort the output dataset chronologically.

    Returns
    -------
    xr.Dataset
        Concatenated dataset with dimension (time, lat, lon).
    """
    base_dir = Path(processed_dir) if processed_dir else PROCESSED_DATA_DIR
    target_dir = base_dir / var_name

    if not target_dir.exists():
        raise FileNotFoundError(f"Processed directory does not exist: {target_dir}")

    files = sorted(list(target_dir.glob(f"{var_name}_*.nc")))
    if not files:
        raise FileNotFoundError(f"No processed files found in {target_dir}")

    datasets = [xr.open_dataset(f) for f in files]
    combined = xr.concat(datasets, dim="time")

    if sort_by_time:
        combined = combined.sortby("time")

    return combined
