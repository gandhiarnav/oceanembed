# OceanEmbed (SIH #26066) - Dataset Summary & Verification Report

**Project Title:** OceanEmbed - Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations  
**Problem Statement ID:** 26066  
**Organization / Department:** Ministry of Earth Sciences (MoES) / INCOIS  
**Dataset Directory:** `/mnt/SharedData/Linux_Shared/SIH_project/Data/dummy_data/dummy`

---

## 1. Executive Summary & SIH Compliance Status

The dummy dataset has been analyzed and verified against the SIH Problem Statement #26066 requirements:

| Parameter | SIH Requirement | Dummy Dataset Value | Status |
| :--- | :--- | :--- | :---: |
| **Spatial Coverage** | North Indian Ocean ($5^\circ\text{N} - 30^\circ\text{N}$, $45^\circ\text{E} - 105^\circ\text{E}$) | $5.00^\circ\text{N} - 30.00^\circ\text{N}$, $45.00^\circ\text{E} - 105.00^\circ\text{E}$ | **PASSED** |
| **Spatial Resolution** | $0.25^\circ \times 0.25^\circ$ | Lat step: $0.25^\circ$, Lon step: $0.25^\circ$ ($101 \times 241$ grid) | **PASSED** |
| **Temporal Resolution** | Daily | 365 days (`2020-01-01` to `2020-12-30`) | **PASSED** |
| **Input Surface Variables** | SST, SSS, SSH/SLA, Surface Currents (U, V), Surface Winds (U, V) | 7 Channels (`SST`, `SSS`, `SSH/SLA`, `U_curr`, `V_curr`, `U_wind`, `V_wind`) | **PASSED** |
| **Target Variables** | Subsurface Temperature at standard depths ($0\text{ m}$ to $1000\text{ m}$) | 15 Depth levels ($0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\text{ m}$) | **PASSED** |
| **Data Integrity** | Clean float32 array, no missing values | $0$ NaNs across all inputs and target tensors | **PASSED** |

---

## 2. File Directory Structure

```
/mnt/SharedData/Linux_Shared/SIH_project/Data/dummy_data/dummy/
├── data-exploration.py        # Python verification script
├── dataset_summary.md         # Summary report (this file)
├── dummy_dates.npy            # Date strings array (shape: 365,)
├── dummy_inputs.npy           # 4D surface input tensor (shape: 365, 7, 101, 241)
├── dummy_targets.npy          # 4D subsurface target tensor (shape: 365, 15, 101, 241)
├── grid_info.npz              # Consolidated grid info NPZ file
└── grid_info/                 # Grid arrays directory
    ├── dates.npy              # Latitudes (shape: 101,)
    ├── depths.npy             # Depths in meters (shape: 15,)
    ├── lat.npy                # Latitudes (shape: 101,)
    └── lon.npy                # Longitudes (shape: 241,)
```

---

## 3. Spatial & Temporal Grid Details

- **Latitude Grid (`lat.npy`):** Shape `(101,)`, Range: `5.0°N` to `30.0°N`, Resolution: `0.25°`
- **Longitude Grid (`lon.npy`):** Shape `(241,)`, Range: `45.0°E` to `105.0°E`, Resolution: `0.25°`
- **Time Coverage (`dates.npy`):** Shape `(365,)`, 365 daily timesteps (1 full year)
- **Vertical Depths (`depths.npy`):** Shape `(15,)`, 15 standard depth levels (meters):  
  `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]`

---

## 4. Input Variables Summary (`dummy_inputs.npy`)

**Shape:** `(365, 7, 101, 241)` | **Data Type:** `float32` | **Memory Size:** ~237.24 MB

| Index | Variable | Description | Unit | Min | Max | Mean | Std | NaN Count |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **SST** | Sea Surface Temperature | $^\circ\text{C}$ | 20.0000 | 33.0000 | 26.5001 | 3.7529 | 0 |
| **1** | **SSS** | Sea Surface Salinity | $\text{PSU}$ | 30.0000 | 38.0000 | 34.0007 | 2.3084 | 0 |
| **2** | **SSH / SLA** | Sea Surface Height / Anomaly | $\text{m}$ | -0.5000 | 0.5000 | 0.0001 | 0.2887 | 0 |
| **3** | **U_curr** | Surface Ocean Current (Zonal) | $\text{m/s}$ | -1.5000 | 1.5000 | -0.0006 | 0.8659 | 0 |
| **4** | **V_curr** | Surface Ocean Current (Meridional) | $\text{m/s}$ | -1.5000 | 1.5000 | -0.0001 | 0.8660 | 0 |
| **5** | **U_wind** | Surface Wind Velocity (Zonal) | $\text{m/s}$ | -15.0000 | 15.0000 | -0.0032 | 8.6607 | 0 |
| **6** | **V_wind** | Surface Wind Velocity (Meridional) | $\text{m/s}$ | -15.0000 | 15.0000 | -0.0005 | 8.6613 | 0 |

---

## 5. Target Subsurface Temperature (`dummy_targets.npy`)

**Shape:** `(365, 15, 101, 241)` | **Data Type:** `float32` | **Memory Size:** ~508.37 MB

| Index | Depth | Min ($^\circ\text{C}$) | Max ($^\circ\text{C}$) | Mean ($^\circ\text{C}$) | Std ($^\circ\text{C}$) | NaN Count | Physical Description |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **0 m** | 25.1571 | 30.5542 | 27.9998 | 0.5002 | 0 | Mixed layer / Surface |
| **1** | **5 m** | 25.1511 | 30.3238 | 27.7610 | 0.5003 | 0 | Upper mixed layer |
| **2** | **10 m** | 24.7723 | 30.2418 | 27.5247 | 0.5001 | 0 | Upper mixed layer |
| **3** | **20 m** | 24.4329 | 29.5950 | 27.0588 | 0.5001 | 0 | Base of mixed layer |
| **4** | **30 m** | 23.9612 | 29.3897 | 26.6026 | 0.5000 | 0 | Upper thermocline |
| **5** | **50 m** | 22.8219 | 28.2386 | 25.7162 | 0.5000 | 0 | Upper thermocline |
| **6** | **75 m** | 22.0580 | 27.3034 | 24.6569 | 0.5000 | 0 | Main thermocline |
| **7** | **100 m** | 21.1033 | 26.1564 | 23.6495 | 0.5002 | 0 | Main thermocline |
| **8** | **125 m** | 19.8144 | 25.3924 | 22.6915 | 0.5000 | 0 | Thermocline |
| **9** | **150 m** | 19.1447 | 24.2852 | 21.7796 | 0.5002 | 0 | Mid thermocline |
| **10** | **200 m** | 17.4705 | 22.6383 | 20.0876 | 0.5001 | 0 | Lower thermocline |
| **11** | **300 m** | 14.5940 | 19.7289 | 17.1713 | 0.5000 | 0 | Subthermocline |
| **12** | **500 m** | 10.1490 | 15.3408 | 12.8289 | 0.4998 | 0 | Intermediate layer |
| **13** | **700 m** | 7.2570 | 12.5268 | 9.9182 | 0.5003 | 0 | Deep water |
| **14** | **1000 m** | 4.6300 | 10.1705 | 7.2483 | 0.5000 | 0 | Deep ocean baseline |

---

## 6. How to Load in Python / PyTorch

```python
import numpy as np
import torch
from torch.utils.data import Dataset

class OceanEmbedDataset(Dataset):
    def __init__(self, data_dir):
        self.inputs = np.load(f"{data_dir}/dummy_inputs.npy")   # (365, 7, 101, 241)
        self.targets = np.load(f"{data_dir}/dummy_targets.npy") # (365, 15, 101, 241)
        self.dates = np.load(f"{data_dir}/dummy_dates.npy")
        
        grid = np.load(f"{data_dir}/grid_info.npz")
        self.lats = grid['lat']
        self.lons = grid['lon']
        self.depths = grid['depths']

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx])   # shape: (7, 101, 241)
        y = torch.from_numpy(self.targets[idx])  # shape: (15, 101, 241)
        return x, y
```
