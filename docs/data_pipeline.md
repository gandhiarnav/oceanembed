Absolutely. Here’s a **copy-paste context block** you can give your CLI agent. I’ll keep it focused on what we’ve actually decided/done, so it can pick up from here.

---

# SIH Project — Data Pipeline Context

## Project

We are working on **Smart India Hackathon (SIH) Problem Statement #26066**.

The project is an ocean ML system that uses **surface ocean/environmental variables to reconstruct/predict subsurface ocean temperature profiles** over the North Indian Ocean.

### Required spatial/temporal domain

```text
Latitude:  5°N → 30°N
Longitude: 45°E → 105°E

Spatial resolution: 0.25° × 0.25°
Temporal resolution: Daily
```

The final spatial grid therefore has:

```text
101 latitude points
241 longitude points
```

because:

```text
(30 - 5) / 0.25 + 1 = 101
(105 - 45) / 0.25 + 1 = 241
```

---

# Team roles

### Person 1 — Data Pipeline & Harmonization — ME

My responsibility:

* acquire datasets
* spatial cropping
* spatial regridding
* temporal alignment
* quality control
* unit conversions
* land masking
* produce final ML tensors

Required **7 input channels**:

```text
0: SST
1: SSS
2: SSH/SLA
3: U current
4: V current
5: U wind
6: V wind
```

Target:

```text
subsurface temperature at 15 depths
```

Depths:

```text
[0, 5, 10, 20, 30, 50, 75, 100,
 125, 150, 200, 300, 500, 700, 1000] meters
```

Expected final shapes:

```text
X = (N, 7, 101, 241)
Y = (N, 15, 101, 241)
```

The exact number of days `N` will depend on the final common temporal coverage.

---

# Dummy dataset

A teammate already created a dummy dataset to establish the data contract.

Current dummy data:

```text
dummy_inputs.npy
shape = (365, 7, 101, 241)
dtype = float32

dummy_targets.npy
shape = (365, 15, 101, 241)
dtype = float32

dummy_dates.npy
shape = (365,)
```

Input channel statistics:

```text
SST       20 → 33 °C
SSS       30 → 38
SSH/SLA   -0.5 → 0.5
U_curr    -1.5 → 1.5
V_curr    -1.5 → 1.5
U_wind    -15 → 15
V_wind    -15 → 15
```

Target temperatures are provided at the 15 depths listed above.


# Overall data architecture

We decided **not to combine raw datasets directly**.

Each dataset is processed individually first:

```text
Raw SST ──→ process ──→ standardized SST
Raw SSS ──→ process ──→ standardized SSS
Raw SSH ──→ process ──→ standardized SSH
Raw currents ──→ process ──→ standardized U/V
Raw winds ──→ process ──→ standardized U/V
```

Each standardized variable should eventually have:

```text
(time, lat, lon)
```

with:

```text
same dates
same 0.25° grid
lat = 101
lon = 241
```

Then combine:

```python
X = np.stack(
    [sst, sss, ssh, u_curr, v_curr, u_wind, v_wind],
    axis=1
)
```

giving:

```text
X = (N, 7, 101, 241)
```

The subsurface target is processed separately:

```text
GLORYS / Argo
       ↓
temperature
       ↓
15 depths
       ↓
same spatial grid
       ↓
same dates
       ↓
Y = (N, 15, 101, 241)
```

Then dates are aligned:

```text
X[2020-01-01] ↔ Y[2020-01-01]
X[2020-01-02] ↔ Y[2020-01-02]
...
```

---

# Current development environment

We are working locally, **not primarily on Kaggle**.

Main directory:

```text
/mnt/SharedData/Linux_Shared/SIH_project/Data
```

We decided to use:

```text
VS Code / local code editor
Python venv
Git
GitHub
```

rather than Kaggle for the data pipeline.

Kaggle may be useful later for model training/GPU if necessary.

---

# Current project structure

Intended structure:

```text
Data/
├── .venv/
│
├── dummy_data/
├── dummy_data.zip
├── NASA_sst_data/
│
├── data/
│   ├── raw/
│   │   ├── sst/
│   │   ├── sss/
│   │   ├── ssh/
│   │   ├── currents/
│   │   ├── winds/
│   │   └── subsurface/
│   │
│   └── processed/
│       ├── sst/
│       ├── sss/
│       ├── ssh/
│       ├── currents/
│       ├── winds/
│       └── subsurface/
│
├── src/
│   ├── download/
│   ├── preprocessing/
│   └── utils/
│
├── notebooks/
├── scripts/
├── requirements.txt
├── .gitignore
└── README.md
```

The existing `NASA_sst_data/` folder has been left alone for now because we are actively using it.

Raw scientific data should **not** be committed to GitHub.

`.gitignore` includes things such as:

```text
.venv/
*.nc
*.nc4
*.h5
*.hdf
*.npy
*.npz
*.zarr/
data/raw/
data/processed/
NASA_sst_data/
__pycache__/
.ipynb_checkpoints/
```

---

# Current SST dataset

For SST we chose:

**CMC 0.1° Global Level-4 SST**

Collection ID:

```text
CMC0.1deg-CMC-L4-GLOB-v3.0
```

It is a daily global SST dataset at approximately:

```text
0.1° × 0.1°
```

We are using it as our first real-data pipeline.

Dataset page:

```text
https://www.earthdata.nasa.gov/data/catalog/pocloud-cmc0.1deg-cmc-l4-glob-v3.0-3.0
```

NASA Earthdata authentication was configured and the PO.DAAC downloader was installed in the venv.

---

# Actual SST data downloaded

We ran:

```bash
podaac-data-downloader \
  -c CMC0.1deg-CMC-L4-GLOB-v3.0 \
  -d ./NASA_sst_data \
  -sd 2020-01-01T00:00:00Z \
  -ed 2020-01-02T00:00:00Z \
  -e .nc
```

It downloaded 3 granules:

```text
20191231120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc
20200101120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc
20200102120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc
```

The file we're currently working with is:

```text
NASA_sst_data/20200101120000-CMC-L4_GHRSST-SSTfnd-CMC0.1deg-GLOB-v02.0-fv03.0.nc
```

Size ~6.7 MB on disk.

There is also:

```text
CMC0.1deg-CMC-L4-GLOB-v3.0.citation.txt
GDS20r5.pdf
```

in that folder.

---

# SST NetCDF structure

We inspected the file with xarray.

Dataset dimensions:

```text
time = 1
lat  = 1801
lon  = 3600
```

Coordinates:

```text
time = 2020-01-01 12:00
lat = -90 → 90 at 0.1°
lon = -180 → 179.9 at 0.1°
```

Important variables:

```text
analysed_sst
analysis_error
sea_ice_fraction
mask
```

The SST variable is:

```text
analysed_sst
```

and its metadata says:

```text
units = kelvin
```

We convert it to Celsius with:

```python
sst_celsius = sst_region - 273.15
sst_celsius.attrs["units"] = "degree_Celsius"
```

---

# SST processing completed so far

## 1. Open NetCDF

Using:

```python
import xarray as xr

ds = xr.open_dataset(file)
```

## 2. Crop to North Indian Ocean

```python
sst_region = sst.sel(
    lat=slice(5, 30),
    lon=slice(45, 105)
)
```

This produced:

```text
(1, 251, 601)
```

because the source resolution is 0.1°:

```text
5 → 30 = 251 points
45 → 105 = 601 points
```

## 3. Convert Kelvin → Celsius

```python
sst_celsius = sst_region - 273.15
sst_celsius.attrs["units"] = "degree_Celsius"
```

## 4. Create target SIH grid

```python
target_lat = np.arange(5, 30.0001, 0.25)
target_lon = np.arange(45, 105.0001, 0.25)
```

giving:

```text
101 latitudes
241 longitudes
```

## 5. Regrid 0.1° → 0.25°

Currently using:

```python
sst_regridded = sst_celsius.interp(
    lat=target_lat,
    lon=target_lon,
    method="linear"
)
```

This produced:

```text
(1, 101, 241)
```

So the spatial harmonization works.

**Note:** We have not yet decided whether `xarray.interp(method="linear")` is the final scientifically preferred regridding method. It is being used for the first working iteration. We may later evaluate xESMF/conservative/bilinear approaches.

---

# SST mask

The CMC file has:

```text
mask
```

with attributes:

```text
long_name:
sea/land/lake/ice field composite mask

flag_masks:
[1, 2, 4, 8, 16]

flag_meanings:
water land optional_lake_surface sea_ice optional_river_surface
```

For our particular 2020-01-01 North Indian Ocean crop, only these values occurred:

```text
1 = water
2 = land
```

We therefore created:

```python
mask_region = mask.sel(
    lat=slice(5, 30),
    lon=slice(45, 105)
)
```

Then regridded the categorical mask with nearest-neighbor:

```python
mask_regridded = mask_region.interp(
    lat=target_lat,
    lon=target_lon,
    method="nearest"
)
```

Then:

```python
ocean_mask = mask_regridded == 1
```

and applied it:

```python
sst_final = sst_regridded.where(ocean_mask)
```

Final result:

```text
shape = (1, 101, 241)
NaN count = 12248
min ≈ 5.97°C
max ≈ 29.55°C
```

We noticed that the minimum is still lower than expected for much of the North Indian Ocean, so this should be investigated later. **Do not assume it is correct just because the mask worked.**

---

# Visualization

We saved a plot of the final SST field.

It visually showed:

* ocean region populated
* land masked white
* smooth SST spatial pattern
* reasonable tropical warming
* no obvious catastrophic regridding problem

The plot looked generally reasonable.

We have **not yet fully resolved coastal/island edge artifacts** or whether the 5.97°C value represents a legitimate ocean pixel, a coastal artifact, or another issue.

---

# Current processed SST output

We successfully saved:

```text
NASA_sst_data/sst_2020-01-01_0.25deg.nc
```

Size approximately:

```text
210 KB
```

It contains the processed:

```text
1 × 101 × 241
```

SST field.

---

# Current scripts

We created:

```text
scripts/inspect_sst.py
scripts/plot_sst.py
```

`inspect_sst.py` currently contains exploratory processing/printing for:

* dataset inspection
* crop
* Celsius conversion
* mask inspection
* target grid
* regridding
* masking
* saving processed SST

It is still an **experimental inspection script**, not yet the final production preprocessing pipeline.

`plot_sst.py` loads:

```text
NASA_sst_data/sst_2020-01-01_0.25deg.nc
```

and saves a PNG because the environment does not have an interactive display.

---

# What we were going to do next

We were about to turn the experimental one-day SST processing into a reusable pipeline:

```text
raw daily CMC files
        ↓
automated crop
        ↓
K → °C
        ↓
regrid
        ↓
mask
        ↓
save processed daily SST
```

Then test it on **3–5 days**, before downloading/processing the entire year.

Only after the 3–5 day pipeline works should we scale to the full required period.

---

# Division of work with friend

I am handling:

```text
SST
data pipeline structure
common 0.25° grid
preprocessing integration
eventual combination of all 7 inputs
```

Friend should investigate:

### SSS

Find a suitable **daily SSS dataset**, preferably SMAP/Copernicus, and report:

```text
dataset name
dataset ID
native resolution
temporal resolution
coverage
2020 availability
variables
units
download method/API
spatial subset capability
```

Desired final:

```text
SSS → daily → 5–30°N, 45–105°E → 0.25°
```

### Subsurface target

Friend should investigate **GLORYS** as the main candidate for the subsurface temperature target.

Need:

```text
dataset name
temperature variable
native horizontal resolution
native temporal resolution
native depth levels
2020 availability
download method/API
whether 15 requested depths can be extracted/interpolated
spatial subset capability
```

Desired target:

```text
GLORYS
 ↓
temperature
 ↓
5–30°N, 45–105°E
 ↓
15 depths:
0,5,10,20,30,50,75,100,
125,150,200,300,500,700,1000 m
 ↓
0.25°
 ↓
daily
```

---

# Remaining 6 input channels

After SST:

```text
SST       → currently being handled
SSS       → friend investigating
SSH/SLA   → still need dataset
U current → still need dataset
V current → still need dataset
U wind    → still need dataset
V wind    → still need dataset
```

Likely categories discussed:

```text
SSH/SLA       → AVISO / Copernicus
Currents      → OSCAR / CMEMS
Winds         → ERA5 / CCMP
```

These are **candidate sources**, not final decisions yet.

---

# Current philosophy

The user wants to move **quickly** and learn only what is necessary.

Do NOT give huge prerequisite tutorials unless asked.

Work **step-by-step**, preferably:

```text
one task
→ run it
→ inspect result
→ explain briefly
→ next task
```

The user explicitly prefers answers to be **a little brief unless they ask otherwise**.

Do not overwhelm them with all the theory at once.

---

# Immediate next task

The next logical task is:

> Refactor the working one-day SST code into a reusable `process_sst.py` pipeline and test it on 3–5 days.

Do **not** immediately download all 365 days.

First prove:

```text
2020-01-01
2020-01-02
2020-01-03
...
```

can be processed automatically into:

```text
(date, 101, 241)
```

and then eventually combined into:

```text
(N, 101, 241)
```

for SST.

The final 7-channel tensor will only be built after each individual source has been independently harmonized.
