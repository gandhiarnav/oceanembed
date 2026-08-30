"""
=============================================================================
OceanEmbed — Day 5: Z-Score Normalization & Tensor Export
=============================================================================
Purpose : Normalize all 7 input channels using z-score normalization,
          export final training tensors as .npy files, and save
          normalization statistics (mean/std) as norm_stats.json
          for de-normalization on model predictions.

Contract:
  Inputs  tensor : (N_times, 7, 101, 241)  [SST, SSS, SSH, U_c, V_c, U_w, V_w]
  Targets tensor : (N_times, 15, 101, 241) [15 depth levels: 0 to 1000m]
  Ocean mask     : (101, 241) boolean
  Norm stats     : JSON with mean/std per channel

Input  : data/processed/surface_inputs_regridded.nc
         data/processed/target_temp_regridded.nc
Output : data/processed/train_inputs.npy
         data/processed/train_targets.npy
         data/processed/ocean_mask.npy
         data/processed/norm_stats.json
=============================================================================
"""

import json
import os
import sys
import numpy as np
import xarray as xr

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Directories ───────────────────────────────────────────────────────────────
PROCESSED_DIR = "./data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ── Canonical 7-Channel Definitions ───────────────────────────────────────────
CANONICAL_CHANNELS = [
    ("SST",    ["sst", "thetao"]),
    ("SSS",    ["sss", "so", "sss_smap"]),
    ("SSH",    ["ssh", "zos"]),
    ("U_curr", ["u_curr", "uo"]),
    ("V_curr", ["v_curr", "vo"]),
    ("U_wind", ["u_wind", "u10"]),
    ("V_wind", ["v_wind", "v10"]),
]

STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

print("=" * 65)
print("  OceanEmbed — Normalization & Tensor Export (7-Channel Pipeline)")
print("=" * 65)

# ── Load processed data ───────────────────────────────────────────────────────
surface_path = os.path.join(PROCESSED_DIR, "surface_inputs_regridded.nc")
target_path  = os.path.join(PROCESSED_DIR, "target_temp_regridded.nc")

if not os.path.exists(surface_path):
    print(f"  ❌ File not found: {surface_path}")
    print("     Run scripts/regrid_harmonize.py first.")
    sys.exit(1)

ds_surf = xr.open_dataset(surface_path)
print(f"  Loaded surface inputs: {surface_path}")
print(f"  Available variables in dataset: {list(ds_surf.data_vars)}")

# ── Build Master Ocean Mask ───────────────────────────────────────────────────
# Find first variable to identify ocean vs land pixels
sample_var_name = list(ds_surf.data_vars)[0]
sample_field = ds_surf[sample_var_name].isel(time=0).values  # (101, 241)
ocean_mask = ~np.isnan(sample_field)  # True = ocean, False = land

mask_path = os.path.join(PROCESSED_DIR, "ocean_mask.npy")
np.save(mask_path, ocean_mask)
print(f"  Ocean mask saved: {mask_path}")
print(f"    Ocean pixels: {ocean_mask.sum()} / {ocean_mask.size} ({ocean_mask.mean() * 100:.1f}%)")
print()

# ── Map Available Variables to Canonical Channels ────────────────────────────
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

# ── STRICT CHANNEL GUARD ──────────────────────────────────────────────────────
# The model contract requires exactly 7 channels.
# Abort early rather than silently producing a wrong-shaped tensor.
if len(active_channels) != len(CANONICAL_CHANNELS):
    missing = [c for c, _ in CANONICAL_CHANNELS if c not in [a for a, _ in active_channels]]
    print("\n  ❌ ABORT: Missing channels detected!")
    print(f"     Required : {len(CANONICAL_CHANNELS)} channels")
    print(f"     Found    : {len(active_channels)} channels")
    print(f"     Missing  : {missing}")
    print("     Ensure all source files are downloaded before running this script:")
    print("       - data/raw/glorys_surface_inputs_jan2020.nc  (SST, SSH, U_curr, V_curr)")
    print("       - data/raw/sss_merged_jan2020.nc             (SSS)")
    print("       - data/raw/era5_winds_jan2020.nc             (U_wind, V_wind)")
    sys.exit(1)

print("  ✅ All 7 required channels found. Proceeding...\n")

n_times = len(ds_surf.time)
n_lat   = len(ds_surf.lat)
n_lon   = len(ds_surf.lon)
n_chan  = len(active_channels)

inputs = np.zeros((n_times, n_chan, n_lat, n_lon), dtype=np.float32)

for ch_idx, (canon_name, var_name) in enumerate(active_channels):
    arr = ds_surf[var_name].values.astype(np.float32)
    inputs[:, ch_idx, :, :] = arr

# ── Z-Score Normalization per Channel (on ocean pixels only) ───────────────────
print("  Normalizing inputs (z-score on ocean pixels) ...")
norm_stats = {}

for ch_idx, (canon_name, var_name) in enumerate(active_channels):
    channel_data = inputs[:, ch_idx, :, :]
    ocean_vals   = channel_data[:, ocean_mask]

    mean_val = float(np.nanmean(ocean_vals))
    std_val  = float(np.nanstd(ocean_vals))
    if std_val < 1e-8 or np.isnan(std_val):
        std_val = 1.0   # avoid divide by zero

    # Normalize
    inputs[:, ch_idx, :, :] = (channel_data - mean_val) / std_val
    # Set land pixels and any remaining NaNs to 0.0
    inputs[:, ch_idx, ~ocean_mask] = 0.0
    inputs[:, ch_idx, :, :] = np.nan_to_num(inputs[:, ch_idx, :, :], nan=0.0)

    norm_stats[canon_name] = {"mean": mean_val, "std": std_val}
    print(f"    {canon_name:8s}: mean = {mean_val:8.4f}, std = {std_val:8.4f}")

# ── Save Norm Stats JSON ──────────────────────────────────────────────────────
stats_path = os.path.join(PROCESSED_DIR, "norm_stats.json")
with open(stats_path, "w") as f:
    json.dump(norm_stats, f, indent=2)
print(f"\n  Saved normalization stats: {stats_path}")

# ── Save Input Tensor ─────────────────────────────────────────────────────────
inputs_path = os.path.join(PROCESSED_DIR, "train_inputs.npy")
np.save(inputs_path, inputs)
print(f"  Saved inputs tensor      : {inputs_path}  shape={inputs.shape} dtype={inputs.dtype}")

# ── Build and Save Target Tensor ──────────────────────────────────────────────
if os.path.exists(target_path):
    print("\n  Building target tensor (GLORYS 3D temperature) ...")
    ds_tgt = xr.open_dataset(target_path)

    tgt_var_name = "thetao" if "thetao" in ds_tgt else list(ds_tgt.data_vars)[0]
    depth_dim = "depth" if "depth" in ds_tgt.dims else "deptht"
    
    targets = ds_tgt[tgt_var_name].values.astype(np.float32)
    # Fill land NaNs with 0.0
    targets = np.nan_to_num(targets, nan=0.0)

    targets_path = os.path.join(PROCESSED_DIR, "train_targets.npy")
    np.save(targets_path, targets)
    print(f"  Saved targets tensor     : {targets_path} shape={targets.shape} dtype={targets.dtype}")
else:
    print(f"\n  ⚠️  Target file not found: {target_path}")

# ── FINAL FORMAT VERIFICATION ────────────────────────────────────────────────
# Confirm output tensors match the dummy data / model contract exactly.
EXPECTED_INPUTS_SHAPE  = (n_times, 7, 101, 241)
EXPECTED_TARGETS_SHAPE = (n_times, 15, 101, 241)

print("\n" + "=" * 65)
print("  FORMAT VERIFICATION (vs SIH-26066 / dummy data contract)")
print("=" * 65)

checks = []

# Inputs
shape_ok = inputs.shape == EXPECTED_INPUTS_SHAPE
dtype_ok = inputs.dtype == np.float32
nan_ok   = np.isnan(inputs).sum() == 0
checks.append(("Inputs shape",  shape_ok,  f"{inputs.shape}  (expected {EXPECTED_INPUTS_SHAPE})"))
checks.append(("Inputs dtype",  dtype_ok,  f"{inputs.dtype}"))
checks.append(("Inputs no NaN", nan_ok,    f"{np.isnan(inputs).sum()} NaNs"))

# Targets
if os.path.exists(target_path):
    tgt_shape_ok = targets.shape == EXPECTED_TARGETS_SHAPE
    tgt_dtype_ok = targets.dtype == np.float32
    tgt_nan_ok   = np.isnan(targets).sum() == 0
    checks.append(("Targets shape",  tgt_shape_ok, f"{targets.shape}  (expected {EXPECTED_TARGETS_SHAPE})"))
    checks.append(("Targets dtype",  tgt_dtype_ok, f"{targets.dtype}"))
    checks.append(("Targets no NaN", tgt_nan_ok,   f"{np.isnan(targets).sum()} NaNs"))

all_passed = True
for label, passed, detail in checks:
    icon = "✅" if passed else "❌"
    print(f"  {icon} {label:<20}: {detail}")
    if not passed:
        all_passed = False

print("=" * 65)
if all_passed:
    print("  ✅ ALL CHECKS PASSED — Output matches model contract!")
else:
    print("  ❌ SOME CHECKS FAILED — Review the issues above before training.")
    sys.exit(1)
print("=" * 65)
