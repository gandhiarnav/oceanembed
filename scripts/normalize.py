"""
=============================================================================
OceanEmbed — Day 5: Z-Score Normalization & Tensor Export
=============================================================================
Purpose : Normalize all 7 input channels using z-score normalization,
          export final training tensors as .npy files, and save
          normalization statistics (mean/std) as norm_stats.json
          for the Viz lead to use for de-normalization on predictions.

Input  : data/processed/surface_inputs_regridded.nc
Output : data/processed/train_inputs.npy   — normalized input tensor
         data/processed/train_targets.npy  — target depth temperature tensor
         data/processed/ocean_mask.npy     — binary ocean/land mask
         data/processed/norm_stats.json    — per-channel mean and std
=============================================================================
"""

import xarray as xr
import numpy as np
import json
import os

# ── Directories ───────────────────────────────────────────────────────────────
PROCESSED_DIR = "./data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Variable → NetCDF variable name mapping ───────────────────────────────────
# Adjust these keys if your GLORYS variable names differ
SURFACE_VAR_MAP = {
    "SST"   : "thetao",   # Sea Surface Temperature (°C)
    "SSH"   : "zos",      # Sea Surface Height (m)
    "U_curr": "uo",       # Eastward current (m/s)
    "V_curr": "vo",       # Northward current (m/s)
    # SSS and winds to be added when ERA5/SMAP data is downloaded
    # "SSS"   : "so",
    # "U_wind": "u10",
    # "V_wind": "v10",
}

CHANNEL_ORDER = ["SST", "SSH", "U_curr", "V_curr"]  # Expand as more data arrives

STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

print("=" * 65)
print("  OceanEmbed — Normalization & Tensor Export")
print("=" * 65)

# ── Load processed data ───────────────────────────────────────────────────────
surface_path = os.path.join(PROCESSED_DIR, "surface_inputs_regridded.nc")
target_path  = os.path.join(PROCESSED_DIR, "target_temp_regridded.nc")

if not os.path.exists(surface_path):
    print(f"  ❌ File not found: {surface_path}")
    print("     Run scripts/regrid_harmonize.py first.")
    exit(1)

ds_surf = xr.open_dataset(surface_path)
print(f"  Loaded surface inputs: {surface_path}")

# ── Build ocean mask ──────────────────────────────────────────────────────────
# True = ocean pixel, False = land pixel
first_var = list(ds_surf.data_vars)[0]
sample_field = ds_surf[first_var].isel(time=0).values  # (101, 241)
ocean_mask = ~np.isnan(sample_field)  # True where data exists (ocean)

mask_path = os.path.join(PROCESSED_DIR, "ocean_mask.npy")
np.save(mask_path, ocean_mask)
print(f"  Ocean mask: {ocean_mask.sum()} ocean pixels / {ocean_mask.size} total")
print(f"  Saved: {mask_path}")
print()

# ── Build multi-channel input array ──────────────────────────────────────────
print("  Building input tensor ...")
n_times = len(ds_surf.time)
n_lat   = len(ds_surf.lat)
n_lon   = len(ds_surf.lon)
n_chan  = len(CHANNEL_ORDER)

inputs = np.zeros((n_times, n_chan, n_lat, n_lon), dtype=np.float32)

for ch_idx, var_name in enumerate(CHANNEL_ORDER):
    nc_name = SURFACE_VAR_MAP[var_name]
    arr = ds_surf[nc_name].values.astype(np.float32)  # (time, lat, lon)
    inputs[:, ch_idx, :, :] = arr
    print(f"    Ch {ch_idx}: {var_name} ({nc_name}) — shape {arr.shape}")

# ── Z-Score normalization ─────────────────────────────────────────────────────
print()
print("  Normalizing inputs (z-score per channel) ...")
norm_stats = {}

for ch_idx, var_name in enumerate(CHANNEL_ORDER):
    # Compute stats only over ocean pixels (ignore land NaN/fill values)
    channel_data = inputs[:, ch_idx, :, :]
    ocean_vals   = channel_data[:, ocean_mask]  # (time, n_ocean_pixels)

    mean_val = float(np.nanmean(ocean_vals))
    std_val  = float(np.nanstd(ocean_vals))
    if std_val < 1e-8:
        std_val = 1.0   # prevent division by zero for constant fields

    # Normalize all pixels (land pixels will be set to 0 after masking)
    inputs[:, ch_idx, :, :] = (channel_data - mean_val) / std_val
    # Set land pixels to 0.0 after normalization
    inputs[:, ch_idx, ocean_mask == False] = 0.0

    norm_stats[var_name] = {"mean": mean_val, "std": std_val}
    print(f"    {var_name:8s}: mean={mean_val:.4f}, std={std_val:.4f}")

# ── Save normalization statistics ─────────────────────────────────────────────
stats_path = os.path.join(PROCESSED_DIR, "norm_stats.json")
with open(stats_path, "w") as f:
    json.dump(norm_stats, f, indent=2)
print(f"\n  Saved normalization stats: {stats_path}")
print("  ⚠️  Share norm_stats.json with Person 3 (Viz) for de-normalization!")

# ── Save input tensor ─────────────────────────────────────────────────────────
inputs_path = os.path.join(PROCESSED_DIR, "train_inputs.npy")
np.save(inputs_path, inputs)
print(f"  Saved inputs tensor : {inputs_path}  shape={inputs.shape}")

# ── Build and save target tensor ──────────────────────────────────────────────
if os.path.exists(target_path):
    print()
    print("  Building target tensor (GLORYS 3D temperature) ...")
    ds_tgt = xr.open_dataset(target_path)

    depth_dim = "depth" if "depth" in ds_tgt.dims else "deptht"
    targets = ds_tgt["thetao"].values.astype(np.float32)
    # Expected shape: (time, 15_depths, 101, 241)
    print(f"    Target shape: {targets.shape}")

    # Fill land NaN in targets with 0.0
    targets = np.nan_to_num(targets, nan=0.0)

    targets_path = os.path.join(PROCESSED_DIR, "train_targets.npy")
    np.save(targets_path, targets)
    print(f"  Saved targets tensor: {targets_path}  shape={targets.shape}")
else:
    print(f"\n  ⚠️  Target file not found: {target_path}")
    print("     Run download_glorys.py and regrid_harmonize.py first.")

print()
print("  ✅ Normalization complete! Hand off to Person 2:")
print(f"     train_inputs.npy   ({inputs.nbytes / 1e6:.1f} MB)")
print(f"     train_targets.npy  (check above)")
print(f"     norm_stats.json    (share with Person 3 too)")
print(f"     ocean_mask.npy     (share with Person 2 for masked loss)")
