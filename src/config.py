"""
Global configuration and spatial/temporal grid specifications for SIH Ocean ML Project.
"""
from pathlib import Path
import numpy as np

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_SST_DIR = RAW_DATA_DIR / "sst"
PROCESSED_SST_DIR = PROCESSED_DATA_DIR / "sst"

# Target domain specification
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

# Subsurface target depth levels (in meters)
TARGET_DEPTHS = [
    0, 5, 10, 20, 30, 50, 75, 100,
    125, 150, 200, 300, 500, 700, 1000
]
