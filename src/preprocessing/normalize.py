"""
OceanEmbed — Normalization & Tensor Export Pipeline.
Performs z-score normalization on all 7 input channels on ocean pixels,
exports final training tensors (.npy), ocean mask, and normalization statistics (norm_stats.json).
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import xarray as xr

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    CANONICAL_CHANNELS,
    TARGET_LAT,
    TARGET_LON,
    TARGET_DEPTHS,
)


def normalize_and_export_tensors(
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Optional[Path] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, dict]:
    """
    Load regridded surface inputs and target datasets, perform z-score normalization,
    and export train_inputs.npy, train_targets.npy, ocean_mask.npy, and norm_stats.json.
    """
    out_dir = Path(output_dir) if output_dir else processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  OceanEmbed — Normalization & Tensor Export (7-Channel Pipeline)")
    print("=" * 65)

    surface_path = processed_dir / "surface_inputs_regridded.nc"
    target_path = processed_dir / "target_temp_regridded.nc"

    if not surface_path.exists():
        raise FileNotFoundError(f"Surface file not found: {surface_path}. Run regrid_harmonize first.")

    ds_surf = xr.open_dataset(surface_path)
    print(f"  Loaded surface inputs: {surface_path}")
    print(f"  Available variables in dataset: {list(ds_surf.data_vars)}")

    # Build Master Ocean Mask
    sample_var_name = list(ds_surf.data_vars)[0]
    sample_field = ds_surf[sample_var_name].isel(time=0).values  # (101, 241)
    ocean_mask = ~np.isnan(sample_field)  # True = ocean, False = land

    mask_path = out_dir / "ocean_mask.npy"
    np.save(mask_path, ocean_mask)
    print(f"  Ocean mask saved: {mask_path}")
    print(f"    Ocean pixels: {ocean_mask.sum()} / {ocean_mask.size} ({ocean_mask.mean() * 100:.1f}%)")
    print()

    # Map Available Variables to Canonical Channels
    active_channels = []
    for canon_name, aliases in CANONICAL_CHANNELS:
        found_var = None
        for alias in aliases:
            if alias in ds_surf.data_vars:
                found_var = alias
                break
        if found_var:
            active_channels.append((canon_name, found_var))
        else:
            print(f"  ⚠️  Channel '{canon_name}' not found in dataset. (Aliases searched: {aliases})")

    print(f"\n  Active channels for export ({len(active_channels)}/{len(CANONICAL_CHANNELS)}):")
    for idx, (cname, vname) in enumerate(active_channels):
        print(f"    Ch {idx}: {cname:8s} (from '{vname}')")
    print()

    if len(active_channels) != len(CANONICAL_CHANNELS):
        missing = [c for c, _ in CANONICAL_CHANNELS if c not in [a for a, _ in active_channels]]
        raise ValueError(
            f"Missing channels: {missing}. Contract requires all {len(CANONICAL_CHANNELS)} channels."
        )

    print("  ✅ All 7 required channels found. Proceeding...\n")

    n_times = len(ds_surf.time)
    n_lat = len(TARGET_LAT)
    n_lon = len(TARGET_LON)
    n_chan = len(active_channels)

    inputs = np.zeros((n_times, n_chan, n_lat, n_lon), dtype=np.float32)

    for ch_idx, (canon_name, var_name) in enumerate(active_channels):
        arr = ds_surf[var_name].values.astype(np.float32)
        inputs[:, ch_idx, :, :] = arr

    # Z-Score Normalization per Channel on ocean pixels
    print("  Normalizing inputs (z-score on ocean pixels)...")
    norm_stats = {}

    for ch_idx, (canon_name, var_name) in enumerate(active_channels):
        channel_data = inputs[:, ch_idx, :, :]
        ocean_vals = channel_data[:, ocean_mask]

        mean_val = float(np.nanmean(ocean_vals))
        std_val = float(np.nanstd(ocean_vals))
        if std_val < 1e-8 or np.isnan(std_val):
            std_val = 1.0

        # Normalize
        inputs[:, ch_idx, :, :] = (channel_data - mean_val) / std_val
        # Set land pixels and any remaining NaNs to 0.0
        inputs[:, ch_idx, ~ocean_mask] = 0.0
        inputs[:, ch_idx, :, :] = np.nan_to_num(inputs[:, ch_idx, :, :], nan=0.0)

        norm_stats[canon_name] = {"mean": mean_val, "std": std_val}
        print(f"    {canon_name:8s}: mean = {mean_val:8.4f}, std = {std_val:8.4f}")

    # Save Norm Stats JSON
    stats_path = out_dir / "norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"\n  Saved normalization stats: {stats_path}")

    # Save Input Tensor
    inputs_path = out_dir / "train_inputs.npy"
    np.save(inputs_path, inputs)
    print(f"  Saved inputs tensor      : {inputs_path}  shape={inputs.shape} dtype={inputs.dtype}")

    # Target Tensor
    targets = None
    if target_path.exists():
        print("\n  Building target tensor (GLORYS 3D temperature)...")
        ds_tgt = xr.open_dataset(target_path)
        tgt_var_name = "thetao" if "thetao" in ds_tgt else list(ds_tgt.data_vars)[0]
        targets = ds_tgt[tgt_var_name].values.astype(np.float32)
        # Fill land NaNs with 0.0
        targets = np.nan_to_num(targets, nan=0.0)

        targets_path = out_dir / "train_targets.npy"
        np.save(targets_path, targets)
        print(f"  Saved targets tensor     : {targets_path} shape={targets.shape} dtype={targets.dtype}")

    # Final Format Verification
    expected_inputs_shape = (n_times, 7, n_lat, n_lon)
    expected_targets_shape = (n_times, len(TARGET_DEPTHS), n_lat, n_lon)

    print("\n" + "=" * 65)
    print("  FORMAT VERIFICATION")
    print("=" * 65)

    assert inputs.shape == expected_inputs_shape, f"Inputs shape mismatch: {inputs.shape} != {expected_inputs_shape}"
    assert inputs.dtype == np.float32, f"Inputs dtype mismatch: {inputs.dtype}"
    assert np.isnan(inputs).sum() == 0, f"Inputs contain NaNs"
    print(f"  ✅ Inputs shape: {inputs.shape}, dtype: {inputs.dtype}, NaNs: 0")

    if targets is not None:
        assert targets.shape == expected_targets_shape, f"Targets shape mismatch: {targets.shape} != {expected_targets_shape}"
        assert targets.dtype == np.float32, f"Targets dtype mismatch: {targets.dtype}"
        assert np.isnan(targets).sum() == 0, f"Targets contain NaNs"
        print(f"  ✅ Targets shape: {targets.shape}, dtype: {targets.dtype}, NaNs: 0")

    print("=" * 65)
    print("  ✅ ALL CHECKS PASSED — Output matches model contract!")
    print("=" * 65)

    return inputs, targets, ocean_mask, norm_stats


if __name__ == "__main__":
    normalize_and_export_tensors()
