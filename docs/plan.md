I need you to PLAN the next phase of our SIH Problem Statement #26066 data pipeline. Do NOT modify any files or execute commands yet.

First inspect the existing repository/codebase, especially:
- src/config.py
- src/preprocessing/regrid_harmonize.py
- src/preprocessing/normalize.py
- src/preprocessing/io.py
- src/download/
- scripts/
- current data/raw and data/processed structure

Context:
We are building a model that predicts subsurface ocean temperature over the North Indian Ocean.

Required domain:
- Latitude: 5°N–30°N
- Longitude: 45°E–105°E
- Resolution: 0.25° × 0.25°
- Grid: 101 × 241
- Daily data

7 surface input channels:
0 SST
1 SSS
2 SSH/SLA
3 U current
4 V current
5 U wind
6 V wind

15 target temperature depths:
[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m

Expected final tensors:
X = (N, 7, 101, 241)
Y = (N, 15, 101, 241)

Current sources:
- SST: CMC 0.1° GHRSST
- SSS: NASA SMAP L3 8-day running mean
- SSH/SLA: CMEMS GLORYS12V1 zos
- U/V currents: CMEMS GLORYS12V1 uo/vo
- U/V winds: ERA5 10m u10/v10
- Target: CMEMS GLORYS12V1 3D thetao

What has already been achieved:
- Centralized configuration is implemented.
- Dummy contract verification passes.
- SST was independently processed and verified.
- The full real-data pipeline has been tested on 2 days.
- All 7 input channels are present.
- Harmonization produces the exact 101 × 241 grid.
- Target extraction produces 15 depth levels.
- Normalization produces NaN-free float32 tensors.
- Current test output:
  X = (2, 7, 101, 241)
  Y = (2, 15, 101, 241)
- ocean_mask.npy contains 12,093 ocean pixels.
- norm_stats.json is generated.
- The pipeline currently reports all format checks as passing.

Important: This is an engineering prototype, so we now need to validate the SCIENTIFIC correctness of the pipeline before scaling to a full month/year of data.

Create a detailed but practical execution plan for the next phase.

The plan MUST prioritize:
1. Auditing each of the 7 channels individually:
   - source
   - variable name
   - units
   - native spatial resolution
   - temporal resolution
   - temporal alignment
   - spatial interpolation/regridding method
   - land/missing-value handling
   - normalization
2. Auditing the GLORYS target:
   - thetao variable
   - depth selection/interpolation
   - temporal alignment
   - spatial regridding
   - masking
3. Checking whether using GLORYS for SSH/currents is scientifically defensible compared with the datasets suggested by the PS (AVISO, OSCAR/CMEMS).
4. Checking for possible data leakage, especially normalization statistics and train/validation/test splitting.
5. Testing the complete pipeline on approximately January 2020 before scaling further.
6. Defining concrete QA checks and acceptance criteria.
7. Only after validation, scaling to the full desired period and producing final training/validation/test datasets.

For every phase, specify:
- what files/code should be inspected or changed
- what command/test should be run
- expected output
- what would constitute a failure
- what should NOT be changed yet

Do not blindly trust the current "ALL CHECKS PASSED" message. Distinguish between:
- shape/format correctness
- numerical correctness
- scientific correctness

Also identify any suspicious assumptions or potential bugs you find in the current implementation.

At the end, give me:
A. The recommended immediate next step
B. The complete ordered roadmap
C. A checklist I can use to track progress

Do not execute anything or modify files. This is a planning/review task only.