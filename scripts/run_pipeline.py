"""
=============================================================================
OceanEmbed — End-to-End Multi-Source Data Pipeline Runner
=============================================================================
Purpose : Single command that runs the entire data pipeline:
            Dummy Contract → Downloads (GLORYS, SMAP SSS, ERA5 Winds)
            → Regrid & Harmonize → Normalize & Export Tensors

Usage:
  python scripts/run_pipeline.py --start 2020-01-01 --end 2020-01-31

Arguments:
  --start          : Start date (YYYY-MM-DD)
  --end            : End date   (YYYY-MM-DD)
  --skip-download  : Skip download steps (use existing raw files)
  --dry-run        : Print steps without executing
=============================================================================
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── Argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="OceanEmbed — End-to-End Data Pipeline"
)
parser.add_argument("--start", type=str, default="2020-01-01",
                    help="Start date (YYYY-MM-DD), default: 2020-01-01")
parser.add_argument("--end", type=str, default="2020-01-31",
                    help="End date (YYYY-MM-DD), default: 2020-01-31")
parser.add_argument("--skip-download", action="store_true",
                    help="Skip download step (use existing raw files)")
parser.add_argument("--dry-run", action="store_true",
                    help="Print steps without executing them")
args = parser.parse_args()

# ── Validate dates ────────────────────────────────────────────────────────────
try:
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt   = datetime.strptime(args.end,   "%Y-%m-%d")
    assert start_dt <= end_dt, "Start date must be before end date"
except (ValueError, AssertionError) as e:
    print(f"  ❌ Invalid date range: {e}")
    sys.exit(1)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(step_name: str, script_path: str, optional: bool = False) -> bool:
    """Run a pipeline step and return True on success."""
    print(f"\n{'─' * 65}")
    print(f"  STEP: {step_name}")
    print(f"{'─' * 65}")
    if args.dry_run:
        print(f"  [DRY RUN] Would run: python {script_path}")
        return True
    
    if not os.path.exists(script_path):
        print(f"  ⚠️  Script not found: {script_path}")
        return optional

    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        if optional:
            print(f"\n  ⚠️  Optional step failed or skipped: {step_name}")
            return True
        else:
            print(f"\n  ❌ Step failed: {step_name}")
            return False
            
    print(f"\n  ✅ Step complete: {step_name}")
    return True


# ── Pipeline banner ───────────────────────────────────────────────────────────
print("=" * 65)
print("  OceanEmbed — End-to-End Multi-Source Data Pipeline")
print("=" * 65)
print(f"  Date range : {args.start} → {args.end}")
print(f"  Mode       : {'DRY RUN' if args.dry_run else 'LIVE'}")
print()

PIPELINE_STEPS = []

# Step 1: Generate dummy data contract (always unblock the team)
PIPELINE_STEPS.append(
    ("Day 1: Generate Dummy Data Contract (7 Channels, 15 Depths)",
     os.path.join(SCRIPTS_DIR, "generate_dummy_data.py"), False)
)

# Step 2: Multi-Source Downloads
if not args.skip_download:
    PIPELINE_STEPS.append(
        ("Download CMEMS GLORYS 3D Target Temperature",
         os.path.join(SCRIPTS_DIR, "download_glorys.py"), False)
    )
    PIPELINE_STEPS.append(
        ("Download CMEMS GLORYS Surface Inputs (SST, SSH, U/V Currents)",
         os.path.join(SCRIPTS_DIR, "download_surface.py"), False)
    )
    PIPELINE_STEPS.append(
        ("Download NASA SMAP Sea Surface Salinity (SSS)",
         os.path.join(SCRIPTS_DIR, "download_sss.py"), True)
    )
    PIPELINE_STEPS.append(
        ("Download ECMWF ERA5 Surface Winds (10m U/V Winds)",
         os.path.join(SCRIPTS_DIR, "download_era5.py"), True)
    )
else:
    print("  ⏭️  Skipping download steps (--skip-download)")

# Step 3: Spatial Harmonization & Multi-Source Regridding
PIPELINE_STEPS.append(
    ("Spatial Harmonization & Multi-Source Regridding (0.25° Grid)",
     os.path.join(SCRIPTS_DIR, "regrid_harmonize.py"), False)
)

# Step 4: Normalization & Final Tensor Export
PIPELINE_STEPS.append(
    ("Z-Score Normalization & Tensor Export",
     os.path.join(SCRIPTS_DIR, "normalize.py"), False)
)

# ── Execute pipeline ──────────────────────────────────────────────────────────
total = len(PIPELINE_STEPS)
for i, (step_name, script_path, optional) in enumerate(PIPELINE_STEPS, 1):
    print(f"\n[{i}/{total}] {step_name}")
    success = run_step(step_name, script_path, optional=optional)
    if not success:
        print(f"\n  ❌ Pipeline aborted at step {i}/{total}: {step_name}")
        sys.exit(1)

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ✅ PIPELINE EXECUTION COMPLETE")
print("=" * 65)
print()
print("  Output files summary:")
outputs = [
    "data/dummy/dummy_inputs.npy",
    "data/dummy/dummy_targets.npy",
    "data/dummy/dummy_dates.npy",
    "data/processed/surface_inputs_regridded.nc",
    "data/processed/target_temp_regridded.nc",
    "data/processed/train_inputs.npy",
    "data/processed/train_targets.npy",
    "data/processed/norm_stats.json",
    "data/processed/ocean_mask.npy",
]
for f in outputs:
    exists = "✅" if os.path.exists(f) else "⚠️  (not yet)"
    size   = f"  ({os.path.getsize(f) / 1e6:.1f} MB)" if os.path.exists(f) else ""
    print(f"  {exists}  {f}{size}")

print()
print("  Artifacts ready for team leads:")
print("    • Person 2 (Model Lead) : train_inputs.npy, train_targets.npy, ocean_mask.npy")
print("    • Person 3 (Viz Lead)   : dummy_inputs.npy, dummy_targets.npy, norm_stats.json")
