# 🌊 OceanEmbed — Satellite Embedding-Based Subsurface Temperature Reconstruction

> **"Use AI to see through the ocean surface — turning everyday satellite snapshots into a full underwater temperature map."**

---

## 📌 Table of Contents

1. [Problem Background](#1-problem-background)
2. [The Core Insight](#2-the-core-insight)
3. [Project Objective](#3-project-objective)
4. [Why It Matters](#4-why-it-matters)
5. [System Architecture Overview](#5-system-architecture-overview)
6. [Input & Output Specification](#6-input--output-specification)
7. [Datasets Required](#7-datasets-required)
8. [Team Roles & Responsibilities](#8-team-roles--responsibilities)
9. [Sprint Strategy](#9-sprint-strategy)
10. [Tech Stack](#10-tech-stack)
11. [Project Structure](#11-project-structure)
12. [Getting Started](#12-getting-started)

---

## 1. Problem Background

Subsurface ocean temperature is a fundamental variable for understanding:
- Ocean circulation and upper-ocean heat content
- Stratification and climate variability
- Air-sea interaction and marine ecosystems
- Marine heatwave monitoring, fisheries management, and data assimilation

### The Measurement Gap

Direct measurements of **subsurface temperature** remain sparse. They rely primarily on in-situ observing systems:

| System | Coverage |
|---|---|
| ARGO Profiling Floats | Sparse point measurements, global but infrequent |
| Moored Buoys | Fixed locations only |
| Gliders | Limited spatial extent |
| Ship Observations | Along specific ship tracks only |

While valuable, these systems are insufficient for generating **continuous, basin-scale subsurface fields**.

### The Satellite Advantage

Satellites provide **continuous, large-scale monitoring** of surface ocean conditions at high spatial and temporal resolution. Surface variables carry **indirect signatures** of subsurface processes through physical mechanisms:

- **Thermocline displacement** → detectable via SSH/SLA
- **Mesoscale eddies** → visible as surface height anomalies and SST fronts
- **Vertical mixing** → expressed as SST changes
- **Ocean-atmosphere coupling** → wind stress and current patterns

---

## 2. The Core Insight

What happens **deep in the ocean** leaves a measurable "fingerprint" on the **surface**:

```
Swirling surface eddy  ──►  Warm/cold water being pushed up or down below
Sea level bump         ──►  Depth of the thermocline (warm/cold interface)
SST fronts             ──►  Subsurface upwelling or downwelling zones
Wind patterns          ──►  Ekman pumping driving vertical mixing
```

> Even though we cannot directly observe underwater, surface observations provide **encodable clues** about the subsurface state. Deep learning can learn this nonlinear mapping.

---

## 3. Project Objective

Develop a **Satellite Embedding-Based Deep Learning Framework** to reconstruct depth-wise subsurface ocean temperature from daily surface satellite observations.

### Domain
- **Region:** North Indian Ocean — **5°N to 30°N, 45°E to 105°E**
- **Spatial Resolution:** 0.25° × 0.25° (101 lat points × 241 lon points)
- **Temporal Resolution:** Daily
- **Target Depths (15 standard levels in meters):**
  `0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000`

### The System Must:

1. Build a preprocessing & harmonization pipeline for multi-source satellite and ocean datasets
2. Standardize all datasets to **0.25° × 0.25° daily** resolution
3. Use **7 surface channels** as model inputs (SST, SSS, SSH, U-current, V-current, U-wind, V-wind)
4. Generate compact **satellite embeddings** using deep learning architectures (CNN, ViT, Autoencoders, GNN, Attention-based hybrids)
5. Train a **reconstruction model** that maps surface state → subsurface temperature profiles
6. Reconstruct **temperature at 15 standard depth levels**
7. Evaluate using **RMSE, Bias, and Correlation** against independent ARGO observations

---

## 4. Why It Matters

A working system provides a **continuous, daily, basin-wide picture** of ocean temperature at all depths, enabling:

| Application | Benefit |
|---|---|
| Marine Heatwave Monitoring | Early warning for coral bleaching events |
| Fisheries Management | Track thermocline depth for fish habitat prediction |
| Ocean Circulation & Climate | Understand heat transport and ENSO teleconnections |
| Data Assimilation | Feed into operational weather and climate forecast models |
| Navy / Maritime Safety | Acoustic propagation modeling, submarine operations |

---

## 5. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    SURFACE INPUTS (7 channels)               │
│   SST │ SSS │ SSH │ U-curr │ V-curr │ U-wind │ V-wind       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              SATELLITE EMBEDDING ENGINE                       │
│   CNN Encoder / Vision Transformer / Autoencoder             │
│   Input: (batch, 7, 101, 241)                                │
│   Output: Compact Latent Vector Z                            │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              RECONSTRUCTION DECODER                           │
│   Depth-conditional decoder / ConvLSTM / 3D CNN             │
│   Output: (batch, 15, 101, 241) — 15 depth temperature maps │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              VALIDATION FRAMEWORK                             │
│   Compare against: GLORYS (training) + ARGO (independent)   │
│   Metrics: RMSE, Bias, R² per depth level                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Input & Output Specification

### Input Tensor Shape
```python
(batch_size, 7, 101, 241)
# 7 channels: SST, SSS, SSH, U_curr, V_curr, U_wind, V_wind
# 101 latitude points:  5°N to 30°N  at 0.25° steps
# 241 longitude points: 45°E to 105°E at 0.25° steps
```

### Output Tensor Shape
```python
(batch_size, 15, 101, 241)
# 15 depth levels: [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m
```

---

## 7. Datasets Required

### 7.1 Training Target (Subsurface Temperature — "Answer Key")

| Dataset | Variable | Source | Notes |
|---|---|---|---|
| **GLORYS12V1** (cmems_mod_glo_phy_my_0.083deg_P1D-m) | 3D Temperature (thetao) | [Copernicus Marine](https://marine.copernicus.eu) | Daily, ~1/12° → regrid to 0.25° |

### 7.2 Surface Input Variables (Model Inputs)

| Variable | Channels | Recommended Dataset | Source |
|---|---|---|---|
| Sea Surface Temperature (SST) | 1 | NOAA OISST v2.1 or GLORYS surface layer | NOAA / CMEMS |
| Sea Surface Salinity (SSS) | 1 | SMAP (NASA) / SMOS (ESA) | NASA Earthdata / CMEMS |
| Sea Surface Height (SSH/SLA) | 1 | CMEMS AVISO Multi-satellite Altimetry | Copernicus Marine |
| Surface Currents U | 1 | OSCAR Currents or GLORYS surface uo | CMEMS / NASA |
| Surface Currents V | 1 | OSCAR Currents or GLORYS surface vo | CMEMS / NASA |
| Surface Wind U | 1 | ERA5 10m U-wind | ECMWF CDS |
| Surface Wind V | 1 | ERA5 10m V-wind | ECMWF CDS |

### 7.3 Validation Dataset (Independent — NOT Used for Training)

| Dataset | Source | Use |
|---|---|---|
| Gridded ARGO | INCOIS Live Access Server (LAS) | Final model accuracy validation |
| Raw ARGO profiles | Argo GDAC (argo.ucsd.edu) | Point-to-grid co-location testing |

> **⚠️ Note:** ARGO data must be kept strictly separate from training. It serves as the independent "ground truth" evaluation.

### 7.4 Practical Shortcut (First Week)
For speed, use **GLORYS12 as a single source** for both inputs and targets:
- Extract its **surface layer** → SST, SSH, U/V currents as inputs
- Extract its **full-depth temperature** → training target
- Use **ARGO** only for final validation (proving generalizability)

---

## 8. Team Roles & Responsibilities

### Role 1: Data Pipeline & Harmonization Lead (Person 1)

**Core Ownership:** Data acquisition, spatial regridding, temporal alignment, quality control, preprocessed tensor generation.

| Task | Details |
|---|---|
| Data Download | Script automated downloads via CMEMS, CDS API, ERDDAP for all 7 surface variables and GLORYS target |
| Spatial Standardization | Regrid all variables to 0.25° × 0.25° using xarray + xESMF/CDO |
| Temporal Alignment | Aggregate/interpolate all datasets to strict daily resolution |
| Land Masking | Create binary ocean mask; fill land pixels with 0 or −999 |
| Normalization | Z-score normalize each channel; save mean/std statistics as JSON |
| Dummy Data (Day 1) | Generate fake numpy tensors for team to unblock immediately |
| Final Deliverable | `train_inputs.npy`, `train_targets.npy`, `norm_stats.json` |

**Tech Stack:** Python, xarray, numpy, netCDF4, xESMF, CDO, scipy

---

### Role 2: Deep Learning & Model Architecture Lead (Person 2)

**Core Ownership:** Neural network design, embedding architecture, training loop, loss functions, evaluation.

| Task | Details |
|---|---|
| Data Loading | PyTorch DataLoader consuming (batch, 7, 101, 241) → (batch, 15, 101, 241) |
| Embedding Engine V0 | Build CNN-based spatial encoder compressing 7 channels into latent Z |
| Reconstruction Decoder | Connect latent Z → 15 depth-level temperature maps |
| Loss Function | Depth-Weighted MSE loss; higher weights for shallower depths |
| Land Mask in Loss | Zero out land pixels in the loss computation |
| Baseline Models | MLP / LightGBM baseline for comparison against the deep learning model |
| Evaluation | Compute RMSE, Bias, R² per depth level against GLORYS and ARGO |
| Final Deliverable | Trained `.pt` weights file + depth-level accuracy report |

**Tech Stack:** PyTorch, scikit-learn, torcheval, NumPy

---

### Role 3: Visualization, Interface & Integration Lead (Person 3)

**Core Ownership:** 3D subsurface rendering, dashboard UI, API integration, demo delivery.

| Task | Details |
|---|---|
| Dashboard Layout | Streamlit/Dash UI with date picker, variable selector, depth slider |
| Surface Heatmaps | 2D maps for all 7 input variables, enforcing NIO bounding box |
| Vertical Cross-sections | Latitude vs. depth slice plots through the ocean |
| Depth Profile Curves | Single-point temperature-vs-depth profile curves (like a sonar profile) |
| 3D Ocean Volume | Interactive 3D rendering of temperature at all depth levels |
| Inference Integration | Connect UI to Person 2's model — date selection triggers forward pass |
| Final Deliverable | Working demo app + presentation slides |

**Tech Stack:** Plotly, Streamlit / Dash, Three.js (if web), FastAPI (inference wrapper)

---

## 9. Sprint Strategy

### The Parallel Prototyping Framework

The key to finishing in 7 days is **decoupling dependencies early** using dummy data.

```
Day 1    Day 2    Day 3    Day 4    Day 5    Day 6    Day 7
  │        │        │        │        │        │        │
  ├──────────────── Person 1: Data Pipeline ─────────────┤
  │ Dummy  │ APIs   │ Regrid │ Mask & │ GLORYS │  QA &  │ Buffer │
  │ tensors│ + DL   │        │ Align  │ target │ Support│        │
  │        │        │        │        │        │        │        │
  ├──────────────── Person 2: Deep Learning ─────────────┤
  │ Setup  │ Arch.  │ Train  │ Tune   │ Eval   │Metrics │ Buffer │
  │        │ + Loss │(dummy) │ (real) │(ARGO)  │ export │        │
  │        │        │        │        │        │        │        │
  ├──────────────── Person 3: Visualization ─────────────┤
  │ UI     │ 3D     │Connect │Polish  │End-to- │ Demo   │ Buffer │
  │ layout │ canvas │ API    │  UI    │  end   │ slides │        │
```

### Critical Rules

| Rule | Why It Matters |
|---|---|
| **Daily 15-min standup** | Day 3 real-data handoff is the #1 failure point — catch slippage early |
| **Define "ship" on Day 1** | Live URL vs. notebook vs. recorded demo changes Days 6–7 completely |
| **Day 6 = hard deadline** | Day 7 is a buffer. Integration always reveals hidden bugs |
| **Dummy data first** | Person 2 and 3 cannot wait 3 days for real data — fake tensors unblock them on Day 1 |

---

## 10. Tech Stack

### 3D Visualization Decision Guide

| Tool | When to Use |
|---|---|
| **Three.js** | Final deliverable is a **live web application** requiring smooth WebGL-based 3D in a browser |
| **Plotly (Python/JS)** | Need **interactive 3D** (scatter/surface) quickly within a Python analytical pipeline |
| **Matplotlib mplot3d** | Need **static 3D plots** for a Jupyter notebook or presentation slides only |

### Full Stack Summary

| Component | Technology |
|---|---|
| Data Wrangling | Python, xarray, numpy, scipy, netCDF4 |
| Regridding | xESMF, CDO (Climate Data Operators) |
| Download APIs | copernicusmarine, cdsapi (ERA5), requests (ERDDAP) |
| Deep Learning | PyTorch, scikit-learn, torcheval |
| Model Export | ONNX / `.pt` (TorchScript) |
| Dashboard | Streamlit or Dash |
| 3D Rendering | Plotly 3D or Three.js |
| API Layer | FastAPI |
| Data Format | NetCDF (.nc), NumPy (.npy), Zarr (.zarr) |

---

## 11. Project Structure

```
OceanEmbed/
│
├── README.md                      ← This file
├── requirements.txt               ← All Python dependencies
│
├── docs/
│   ├── phase1_plan.md             ← Detailed Phase 1 sprint plan
│   ├── data_pipeline.md           ← Person 1 day-by-day guide
│   ├── model_architecture.md      ← Person 2 architecture guide
│   └── viz_dashboard.md           ← Person 3 UI guide
│
├── data/
│   ├── raw/                       ← Downloaded .nc files (gitignored)
│   ├── processed/                 ← Regridded, masked, normalized data
│   └── dummy/                     ← Fake tensors for team unblocking
│
├── scripts/
│   ├── download_glorys.py         ← Download GLORYS target (3D temp)
│   ├── download_surface.py        ← Download surface input variables
│   ├── generate_dummy_data.py     ← Day 1 dummy tensor contract
│   ├── regrid_harmonize.py        ← Spatial + temporal standardization
│   ├── normalize.py               ← Z-score normalization + stats export
│   └── run_pipeline.py            ← End-to-end pipeline runner
│
├── model/
│   ├── dataloader.py              ← PyTorch Dataset + DataLoader
│   ├── encoder.py                 ← CNN Embedding Engine
│   ├── decoder.py                 ← Depth reconstruction head
│   ├── loss.py                    ← Depth-weighted MSE loss
│   └── train.py                   ← Training loop
│
├── validation/
│   ├── collocate_argo.py          ← ARGO point-to-grid matching
│   └── metrics.py                 ← RMSE, Bias, R² calculators
│
└── viz/
    ├── app.py                     ← Streamlit/Dash dashboard
    ├── surface_maps.py            ← 2D surface variable heatmaps
    └── subsurface_3d.py           ← 3D depth visualization
```

---

## 12. Getting Started

### Prerequisites
```bash
pip install copernicusmarine xarray numpy scipy netCDF4 torch plotly streamlit
```

### Step 1 — Authenticate with Copernicus Marine
```bash
# Register free at https://marine.copernicus.eu
pip install copernicusmarine
copernicusmarine login
```

### Step 2 — Generate Dummy Data (Day 1 Priority)
```bash
python scripts/generate_dummy_data.py
```

### Step 3 — Download Real Data
```bash
python scripts/download_glorys.py      # 3D temperature target
python scripts/download_surface.py     # Surface input variables
```

### Step 4 — Run Full Pipeline
```bash
python scripts/run_pipeline.py --start 2020-01-01 --end 2020-12-31
```

### Step 5 — Train Model
```bash
python model/train.py --epochs 50 --batch_size 16
```

### Step 6 — Launch Dashboard
```bash
streamlit run viz/app.py
```

---

*OceanEmbed — Built for the North Indian Ocean Subsurface Temperature Reconstruction Challenge*
