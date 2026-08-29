"""
=============================================================================
OceanEmbed — Day 1: Dummy Data Contract Generator
=============================================================================
Purpose:  Generate fake numpy tensors with the EXACT shapes the real data
          will have, so Person 2 (Model) and Person 3 (Viz) can start
          building their systems immediately without waiting for real data.

Tensor Contract:
  Input  shape: (batch_size, 7,  101, 241)  — 7 surface channels
  Output shape: (batch_size, 15, 101, 241)  — 15 depth temperature levels

Channel Order (inputs):
  0: SST   — Sea Surface Temperature      (°C)
  1: SSS   — Sea Surface Salinity         (PSU)
  2: SSH   — Sea Surface Height           (m)
  3: U_c   — Surface Current U-component  (m/s)
  4: V_c   — Surface Current V-component  (m/s)
  5: U_w   — Surface Wind U-component     (m/s)
  6: V_w   — Surface Wind V-component     (m/s)

Depth Levels (outputs, in meters):
  [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

Grid:
  Latitude : 5.0°N to 30.0°N  at 0.25° → 101 points
  Longitude: 45.0°E to 105.0°E at 0.25° → 241 points

Run:
  python scripts/generate_dummy_data.py
=============================================================================
"""

import numpy as np
import os
import pandas as pd

# ── Grid Definition ───────────────────────────────────────────────────────────
LAT   = np.arange(5.0, 30.25, 0.25)     # 101 latitude points
LON   = np.arange(45.0, 105.25, 0.25)   # 241 longitude points
DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

N_LAT     = len(LAT)      # 101
N_LON     = len(LON)      # 241
N_CHANNELS = 7            # input channels
N_DEPTHS  = len(DEPTHS)   # 15

BATCH_SIZE = 365           # one sample per day of 2020 (matches real data structure)

# ── Variable metadata (for documentation/color scaling) ──────────────────────
VARIABLE_INFO = {
    0: {"name": "SST",   "unit": "°C",   "min": 20.0,  "max": 33.0},
    1: {"name": "SSS",   "unit": "PSU",  "min": 30.0,  "max": 38.0},
    2: {"name": "SSH",   "unit": "m",    "min": -0.5,  "max": 0.5},
    3: {"name": "U_curr","unit": "m/s",  "min": -1.5,  "max": 1.5},
    4: {"name": "V_curr","unit": "m/s",  "min": -1.5,  "max": 1.5},
    5: {"name": "U_wind","unit": "m/s",  "min": -15.0, "max": 15.0},
    6: {"name": "V_wind","unit": "m/s",  "min": -15.0, "max": 15.0},
}

# ── Output Directory ─────────────────────────────────────────────────────────
OUTPUT_DIR = "./data/dummy"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  OceanEmbed — Day 1 Dummy Data Contract Generator")
print("=" * 65)
print(f"\n  Grid         : {N_LAT} lat × {N_LON} lon (0.25° resolution)")
print(f"  Input shape  : ({BATCH_SIZE}, {N_CHANNELS}, {N_LAT}, {N_LON})")
print(f"  Output shape : ({BATCH_SIZE}, {N_DEPTHS}, {N_LAT}, {N_LON})")
print(f"  Channels     : {[v['name'] for v in VARIABLE_INFO.values()]}")
print(f"  Depths (m)   : {DEPTHS}")
print()

# ── Generate Dummy Input Tensor (surface variables) ───────────────────────────
np.random.seed(42)  # Reproducible for all team members

dummy_inputs = np.zeros((BATCH_SIZE, N_CHANNELS, N_LAT, N_LON), dtype=np.float32)

for ch, info in VARIABLE_INFO.items():
    # Use realistic value ranges per variable (not just 0-1)
    lo, hi = info["min"], info["max"]
    dummy_inputs[:, ch, :, :] = np.random.uniform(lo, hi, size=(BATCH_SIZE, N_LAT, N_LON)).astype(np.float32)

# ── Generate Dummy Target Tensor (subsurface temperature) ─────────────────────
# Realistic ocean temperature: warm at surface (~28°C), cold at depth (~4°C)
dummy_targets = np.zeros((BATCH_SIZE, N_DEPTHS, N_LAT, N_LON), dtype=np.float32)

for d_idx, depth in enumerate(DEPTHS):
    # Simple exponential decay from surface warm to deep cold
    surface_temp = 28.0
    deep_temp    = 4.0
    decay_scale  = 500.0  # e-folding scale in meters
    base_temp = deep_temp + (surface_temp - deep_temp) * np.exp(-depth / decay_scale)
    noise     = np.random.randn(BATCH_SIZE, N_LAT, N_LON).astype(np.float32) * 0.5
    dummy_targets[:, d_idx, :, :] = base_temp + noise

# ── Save tensors ──────────────────────────────────────────────────────────────
inputs_path  = os.path.join(OUTPUT_DIR, "dummy_inputs.npy")
targets_path = os.path.join(OUTPUT_DIR, "dummy_targets.npy")
grid_path    = os.path.join(OUTPUT_DIR, "grid_info.npz")

# ── Dates array (one per sample — 2020 full year) ───────────────────────────
dates = pd.date_range(start="2020-01-01", periods=BATCH_SIZE, freq="D")
dates_str = np.array([str(d.date()) for d in dates])  # e.g. ["2020-01-01", ...]

np.save(inputs_path,  dummy_inputs)
np.save(targets_path, dummy_targets)
np.savez(grid_path, lat=LAT, lon=LON, depths=np.array(DEPTHS), dates=dates_str)

# Also save dates separately for easy loading
dates_path = os.path.join(OUTPUT_DIR, "dummy_dates.npy")
np.save(dates_path, dates_str)

# ── Verification printout ─────────────────────────────────────────────────────
print("  Generated tensors:")
print(f"    dummy_inputs  : shape={dummy_inputs.shape}, dtype={dummy_inputs.dtype}")
print(f"    dummy_targets : shape={dummy_targets.shape}, dtype={dummy_targets.dtype}")
print()
print("  Channel value ranges:")
for ch, info in VARIABLE_INFO.items():
    ch_min = dummy_inputs[:, ch].min()
    ch_max = dummy_inputs[:, ch].max()
    print(f"    Ch {ch} ({info['name']:8s}): [{ch_min:.2f}, {ch_max:.2f}] {info['unit']}")
print()
print("  Target depth profiles (mean temperature at each level):")
for d_idx, depth in enumerate(DEPTHS):
    mean_t = dummy_targets[:, d_idx].mean()
    print(f"    {depth:>4d} m  →  {mean_t:.2f} °C")

print()
print(f"  ✅ Saved to: {OUTPUT_DIR}/")
print(f"     dummy_inputs.npy  ({dummy_inputs.nbytes / 1e6:.1f} MB)")
print(f"     dummy_targets.npy ({dummy_targets.nbytes / 1e6:.1f} MB)")
print(f"     dummy_dates.npy   ({len(dates_str)} dates: {dates_str[0]} → {dates_str[-1]})")
print(f"     grid_info.npz     (lat, lon, depths, dates arrays)")
print()
print("  Share the dummy/ folder with Person 2 (Model) and Person 3 (Viz) now.")
print()
print("  Usage example for teammates:")
print("    inputs  = np.load('dummy_inputs.npy')   # (365, 7, 101, 241)")
print("    targets = np.load('dummy_targets.npy')  # (365, 15, 101, 241)")
print("    dates   = np.load('dummy_dates.npy')    # ['2020-01-01', ..., '2020-12-31']")
print("    # To get data for Jan 5th:")
print("    idx = np.where(dates == '2020-01-05')[0][0]")
print("    day_data = inputs[idx]  # shape (7, 101, 241)")
