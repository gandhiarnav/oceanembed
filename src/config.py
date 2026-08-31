"""
Global configuration and spatial/temporal grid specifications for SIH Ocean ML Project (OceanEmbed).
"""
from pathlib import Path
import numpy as np

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DUMMY_DATA_DIR = PROJECT_ROOT / "dummy_data" / "dummy"

# Per-variable raw & processed directories
RAW_SST_DIR = RAW_DATA_DIR / "sst"
RAW_SSS_DIR = RAW_DATA_DIR / "sss"
RAW_WINDS_DIR = RAW_DATA_DIR / "winds"
RAW_SURFACE_DIR = RAW_DATA_DIR / "surface"
RAW_SUBSURFACE_DIR = RAW_DATA_DIR / "subsurface"

PROCESSED_SST_DIR = PROCESSED_DATA_DIR / "sst"
PROCESSED_SSS_DIR = PROCESSED_DATA_DIR / "sss"
PROCESSED_WINDS_DIR = PROCESSED_DATA_DIR / "winds"

# Target domain specification (North Indian Ocean)
LAT_MIN = 5.0
LAT_MAX = 30.0
LAT_STEP = 0.25
NUM_LAT = int(round((LAT_MAX - LAT_MIN) / LAT_STEP)) + 1  # 101

LON_MIN = 45.0
LON_MAX = 105.0
LON_STEP = 0.25
NUM_LON = int(round((LON_MAX - LON_MIN) / LON_STEP)) + 1  # 241

# Standard target coordinates
TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, NUM_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, NUM_LON)

# Subsurface target depth levels (15 levels in meters)
TARGET_DEPTHS = [
    0, 5, 10, 20, 30, 50, 75, 100,
    125, 150, 200, 300, 500, 700, 1000
]

# Canonical 7 input channels and their alias mappings across datasets
CANONICAL_CHANNELS = [
    ("SST",    ["sst", "thetao", "analysed_sst"]),
    ("SSS",    ["sss", "so", "sss_smap"]),
    ("SSH",    ["ssh", "zos", "sla"]),
    ("U_curr", ["u_curr", "uo"]),
    ("V_curr", ["v_curr", "vo"]),
    ("U_wind", ["u_wind", "u10"]),
    ("V_wind", ["v_wind", "v10"]),
]

CHANNEL_NAMES = [name for name, _ in CANONICAL_CHANNELS]

# Default temporal train / validation / test splits
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
