# Phase 1: Data Harmonization & End-to-End Pipeline Integration

## Objective

The primary goal of Phase 1 is to **eliminate integration bottlenecks** by building a complete, functioning system using dummy data, while simultaneously executing the automated download and regridding of the real satellite datasets.

By the end of Phase 1, the following must be fully connected and operational:
- ✅ Preprocessing pipeline
- ✅ Deep learning architecture (forward pass)
- ✅ Visualization frontend (rendering dummy outputs)

---

## Grid Specification (Fixed for the Entire Project)

| Parameter | Value |
|---|---|
| Region | North Indian Ocean |
| Latitude Range | 5°N to 30°N |
| Longitude Range | 45°E to 105°E |
| Spatial Resolution | 0.25° × 0.25° |
| Latitude Points | **101** |
| Longitude Points | **241** |
| Temporal Resolution | Daily |
| Input Channels | **7** (SST, SSS, SSH, U-curr, V-curr, U-wind, V-wind) |
| Output Depth Levels | **15** (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m) |

### Canonical Tensor Shapes
```python
Input  shape: (batch_size, 7,  101, 241)   # Surface satellite observations
Output shape: (batch_size, 15, 101, 241)   # Subsurface temperature at 15 depths
```

---

## Work Breakdown by Role

### Role 1 — Data Pipeline & Harmonization

**Owner:** Person 1 | **Stack:** Python, xarray, numpy, netCDF4, xESMF

| Task | Execution Details | Success Criteria |
|---|---|---|
| **Dummy Data Contract** | Generate random 4D numpy arrays: `(batch, 7, 101, 241)` inputs and `(batch, 15, 101, 241)` targets | Model and Viz leads have local access on Day 1 |
| **Data Acquisition** | Script automated downloads via CMEMS API for SST, SSS, SSH, Currents, Winds, and GLORYS target | Scripts pull 1-month continuous test window without errors |
| **Spatial Standardization** | Regrid all variables to strict 0.25°×0.25° grid using xarray + xESMF/CDO | All 7 surface variables align without coordinate mismatches |
| **Temporal Alignment** | Aggregate/interpolate all datasets to strict daily resolution | No missing days; NaN values handled via interpolation or masking |
| **GLORYS Target Prep** | Extract temperature at the 15 mandated depth levels, apply same bounding box and land mask | Clean `train_targets.npy` ready for model training |
| **Normalization** | Z-score normalize each channel; export `norm_stats.json` | Model Lead and Viz Lead receive normalization statistics |

---

### Role 2 — Satellite Embedding & Baseline Model

**Owner:** Person 2 | **Stack:** PyTorch, scikit-learn, torcheval

| Task | Execution Details | Success Criteria |
|---|---|---|
| **I/O Wiring** | Configure PyTorch DataLoader to ingest `(batch, 7, 101, 241)` → `(batch, 15, 101, 241)` | DataLoader yields batches without memory leaks |
| **Embedding Engine V0** | Build baseline CNN encoder compressing 7 input channels into compact latent space | Forward pass runs on dummy tensor; network compiles |
| **Reconstruction Head** | Connect latent vector to a decoder outputting 15 depth temperature maps | Depth-Weighted MSE loss computes; backpropagation updates weights |
| **Evaluation Framework** | Script metric calculators: RMSE (overall + depth-wise), R², Bias | Metrics output correctly for dummy predictions |

---

### Role 3 — Visualization & Validation UI

**Owner:** Person 3 | **Stack:** Streamlit/Dash, Plotly, Three.js

| Task | Execution Details | Success Criteria |
|---|---|---|
| **Dashboard Layout** | Build UI shell with date pickers, variable selectors, depth sliders | UI runs locally; layout is responsive |
| **Surface Heatmaps** | 2D heatmaps for all 7 input variables over NIO bounding box | Maps update dynamically from dummy data selections |
| **Subsurface 3D Rendering** | Vertical slice (lat vs. depth) and temperature-vs-depth profile curves | 3D visuals render without UI lag |
| **API/Inference Link** | Connect UI to Person 2's inference script | "Predict" button triggers forward pass; 3D map updates with result |

---

## Day-by-Day Execution Plan (Person 1 — Data Lead)

### Day 1: The Data Contract & Dummy Generation
**Goal: Unblock the entire team. Do this before anything else.**

1. **Define the exact grid:**
   - Latitudes: `np.arange(5.0, 30.25, 0.25)` → 101 points
   - Longitudes: `np.arange(45.0, 105.25, 0.25)` → 241 points
   - Depths: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]`

2. **Build dummy input tensor:**
   ```python
   dummy_inputs = np.random.randn(100, 7, 101, 241).astype(np.float32)
   # 100 samples, 7 channels, 101 lat, 241 lon
   ```

3. **Build dummy target tensor:**
   ```python
   dummy_targets = np.random.randn(100, 15, 101, 241).astype(np.float32)
   # 100 samples, 15 depth levels, 101 lat, 241 lon
   ```

4. **Save and share:**
   ```bash
   numpy.save("data/dummy/dummy_inputs.npy", dummy_inputs)
   numpy.save("data/dummy/dummy_targets.npy", dummy_targets)
   ```

**✅ Deliverable:** `dummy_inputs.npy` and `dummy_targets.npy` pushed to shared repo before end of Day 1.

---

### Day 2: Automated Ingestion & API Scripting
**Goal: Write download scripts for all real historical data. Target: 2020 (1-year sample).**

1. **Copernicus Marine (CMEMS):**
   - Script `download_glorys.py`: Pull daily GLORYS12V1 3D temperature (thetao) — your training target
   - Script `download_surface.py`: Pull surface SST (thetao), SSH (zos), U/V currents (uo, vo) from same GLORYS dataset
   - Authenticate: `copernicusmarine login`

2. **ECMWF CDS API (ERA5 Winds):**
   - Script `download_era5.py`: Pull daily surface winds (U/V at 10m) for 2020

3. **SSS (Salinity — if time permits):**
   - Script `download_sss.py`: SMAP or SMOS Level-4 gridded salinity
   - Note: May require interpolation due to coarser native resolution

**✅ Deliverable:** Python scripts that successfully download raw `.nc` files into `data/raw/` folder.

---

### Day 3: Spatial Harmonization & Regridding
**Goal: Force all datasets onto the exact same 101×241 coordinate system.**

This is the most technically challenging day. Different satellite products come on different grids.

1. **Define master target grid:**
   ```python
   import xarray as xr
   import numpy as np
   lat_target = np.arange(5.0, 30.25, 0.25)   # 101 points
   lon_target = np.arange(45.0, 105.25, 0.25)  # 241 points
   ```

2. **Crop all datasets** to the NIO bounding box first (faster processing)

3. **Regrid using xESMF** (bilinear interpolation):
   ```python
   import xesmf as xe
   ds_out = xr.Dataset({"lat": lat_target, "lon": lon_target})
   regridder = xe.Regridder(ds_in, ds_out, "bilinear")
   ds_regridded = regridder(ds_in)
   ```

4. **Verify alignment**: Check that lat/lon coordinates of all 7 variables match exactly — no off-by-one pixel shifts

**✅ Deliverable:** A unified `xarray.Dataset` containing all 7 input variables, perfectly aligned in space.

---

### Day 4: Temporal Alignment & Land Masking
**Goal: Clean, gap-free, daily input files with NaN values handled.**

1. **Daily temporal aggregation:**
   ```python
   ds_daily = ds.resample(time="1D").mean()
   ```

2. **Create the Master Land Mask:**
   ```python
   # Use any variable that has NaN over land (e.g., SST)
   land_mask = ~np.isnan(sst_sample)  # True = ocean, False = land
   np.save("data/processed/ocean_mask.npy", land_mask)
   ```

3. **Fill land pixels:**
   ```python
   # Fill with 0.0 (model will mask these out during training)
   ds_filled = ds.fillna(0.0)
   ```

4. **Handle missing days:** Use forward-fill or linear interpolation for isolated missing days:
   ```python
   ds_filled = ds_daily.interpolate_na(dim="time", method="linear")
   ```

**✅ Deliverable:** Clean, gap-free, land-masked daily NetCDF files ready for normalization.

---

### Day 5: Target Variable Preparation & Normalization
**Goal: Extract GLORYS "answer key" at 15 depth levels + normalize all inputs.**

1. **Extract GLORYS at mandated depths:**
   ```python
   STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
   ds_target = glorys_ds.sel(depth=STANDARD_DEPTHS, method="nearest")
   # Apply same bounding box and land mask
   ```

2. **Z-Score normalization per channel:**
   ```python
   import json
   norm_stats = {}
   for i, var_name in enumerate(["sst", "sss", "ssh", "u_curr", "v_curr", "u_wind", "v_wind"]):
       mean_val = float(inputs[:, i].mean())
       std_val  = float(inputs[:, i].std())
       inputs[:, i] = (inputs[:, i] - mean_val) / std_val
       norm_stats[var_name] = {"mean": mean_val, "std": std_val}
   with open("data/processed/norm_stats.json", "w") as f:
       json.dump(norm_stats, f, indent=2)
   ```

**✅ Deliverable:** `train_inputs.npy`, `train_targets.npy`, `norm_stats.json` handed to Person 2.

---

### Day 6: ARGO Validation Data — Point-to-Grid Co-location
**Goal: Build the script that links floating ARGO sensors to your grid cells for validation.**

ARGO floats report at irregular lat/lon/time positions. You must match each float profile to the nearest model grid cell.

1. **Co-location algorithm:**
   ```python
   def collocate_argo_profile(argo_lat, argo_lon, argo_date, model_ds):
       """Find nearest grid cell and date to an ARGO profile."""
       lat_idx = np.argmin(np.abs(model_ds.lat.values - argo_lat))
       lon_idx = np.argmin(np.abs(model_ds.lon.values - argo_lon))
       return model_ds.isel(lat=lat_idx, lon=lon_idx).sel(time=argo_date, method="nearest")
   ```

2. **Interpolate ARGO depths** to your 15 standard levels:
   ```python
   from scipy.interpolate import interp1d
   f = interp1d(argo_depths, argo_temps, bounds_error=False, fill_value="extrapolate")
   argo_at_standard_depths = f(STANDARD_DEPTHS)
   ```

3. **Export as CSV:**
   - Columns: `date, lat, lon, depth, argo_temp, glorys_temp, model_prediction`

**✅ Deliverable:** `validation/argo_collocated.csv` handed to Person 2 for metric calculation.

---

### Day 7: Pipeline Automation & QA
**Goal: Wrap everything into a single executable pipeline with quality checks.**

1. **Create `run_pipeline.py`:**
   ```bash
   python scripts/run_pipeline.py --start 2020-01-01 --end 2020-12-31
   # Downloads → Regrids → Masks → Normalizes → Exports tensors
   ```

2. **Data Quality Checks:**
   - ✅ Verify land mask hasn't shifted (Sri Lanka should be fully masked)
   - ✅ Check for any remaining NaN values in output tensors
   - ✅ Spot-check a random day: plot all 7 channels, verify coastlines look correct
   - ✅ Confirm depth extraction: plot temperature profile at a known ocean point

3. **Documentation:**
   - Write `docs/data_pipeline.md` with all download commands and known issues

**✅ Deliverable:** One-command pipeline + QA report confirming data integrity.

---

## Dependency & Handoff Schedule

```
Person 1                    Person 2                Person 3
─────────                   ─────────               ─────────
Day 1: dummy_inputs.npy ──► DataLoader setup        UI layout
       dummy_targets.npy ──► Architecture V0 ──────► Dummy rendering

Day 3: real data arrives ──► Switch to real training ► Swap in real data

Day 5: norm_stats.json ───► Normalization-aware ───► Viz normalization
       train_inputs.npy      model training           color scales

Day 6: argo_collocated ───► Validation metrics      Validation view
        .csv

Day 7: QA sign-off ───────► Final trained weights ──► Integrated demo
```

---

## Known Technical Challenges

| Challenge | Mitigation |
|---|---|
| GLORYS native resolution ~1/12° ≠ target 0.25° | Use bilinear regridding with xESMF |
| SSS native resolution 25–60 km (coarser) | Bicubic or Gaussian interpolation to 0.25° |
| ARGO floats are sparse irregular points | Nearest-neighbor co-location + linear depth interpolation |
| Land pixels = NaN in all marine datasets | Create ocean mask on Day 4; fill land with 0.0 |
| Large file sizes (global GLORYS files are multi-GB) | Always use `copernicusmarine.subset()` to crop before download |
| API download failures / timeouts | Wrap downloads in retry loops with exponential backoff |

---

## Evaluation Metrics

All metrics computed **per depth level** and **overall averaged**:

| Metric | Formula | Target |
|---|---|---|
| **RMSE** | √(mean((pred - true)²)) | Minimize |
| **Bias** | mean(pred - true) | Near zero |
| **R² (Correlation)** | 1 - SS_res/SS_tot | Maximize (→ 1.0) |
| **Depth-wise RMSE** | RMSE at each of 15 depth levels | Plotted as vertical profile |

---

*Phase 1 Plan — OceanEmbed Project | Data Harmonization & End-to-End Pipeline Integration*
