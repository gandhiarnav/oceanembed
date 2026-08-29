"""
=============================================================================
OceanEmbed — Day 7: End-to-End Pipeline Runner
=============================================================================
Purpose : Single command that runs the entire data pipeline:
            Download → Regrid → Mask → Normalize → Export tensors

Usage:
  python scripts/run_pipeline.py --start 2020-01-01 --end 2020-01-31

Arguments:
  --start   : Start date (YYYY-MM-DD)
  --end     : End date   (YYYY-MM-DD)
  --skip-download : Skip download step (use existing raw files)
  --dry-run       : Print steps without executing

Output:
  data/raw/         ← Downloaded NetCDF files
  data/processed/   ← Regridded, masked, normalized tensors
  data/processed/train_inputs.npy
  data/processed/train_targets.npy
  data/processed/norm_stats.json
  data/processed/ocean_mask.npy
=============================================================================
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime

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


def run_step(step_name: str, script_path: str) -> bool:
    """Run a pipeline step and return True on success."""
    print(f"\n{'─' * 65}")
    print(f"  STEP: {step_name}")
    print(f"{'─' * 65}")
    if args.dry_run:
        print(f"  [DRY RUN] Would run: python {script_path}")
        return True
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        print(f"\n  ❌ Step failed: {step_name}")
        return False
    print(f"\n  ✅ Step complete: {step_name}")
    return True


# ── Pipeline banner ───────────────────────────────────────────────────────────
print("=" * 65)
print("  OceanEmbed — End-to-End Data Pipeline")
print("=" * 65)
print(f"  Date range : {args.start} → {args.end}")
print(f"  Mode       : {'DRY RUN' if args.dry_run else 'LIVE'}")
print()

PIPELINE_STEPS = []

# Step 1: Generate dummy data (always run first to unblock team)
PIPELINE_STEPS.append(
    ("Day 1: Generate Dummy Data Contract",
     os.path.join(SCRIPTS_DIR, "generate_dummy_data.py"))
)

# Step 2 & 3: Downloads (skip if --skip-download)
if not args.skip_download:
    PIPELINE_STEPS.append(
        ("Day 2: Download GLORYS 3D Target Temperature",
         os.path.join(SCRIPTS_DIR, "download_glorys.py"))
    )
    PIPELINE_STEPS.append(
        ("Day 2: Download GLORYS Surface Input Variables",
         os.path.join(SCRIPTS_DIR, "download_surface.py"))
    )
else:
    print("  ⏭️  Skipping download steps (--skip-download)")

# Step 4: Regridding
PIPELINE_STEPS.append(
    ("Day 3: Spatial Harmonization & Regridding",
     os.path.join(SCRIPTS_DIR, "regrid_harmonize.py"))
)

# Step 5: Normalization & tensor export
PIPELINE_STEPS.append(
    ("Day 5: Z-Score Normalization & Tensor Export",
     os.path.join(SCRIPTS_DIR, "normalize.py"))
)

# ── Execute pipeline ──────────────────────────────────────────────────────────
total = len(PIPELINE_STEPS)
for i, (step_name, script_path) in enumerate(PIPELINE_STEPS, 1):
    print(f"\n[{i}/{total}] {step_name}")
    success = run_step(step_name, script_path)
    if not success:
        print(f"\n  ❌ Pipeline aborted at step {i}/{total}: {step_name}")
        sys.exit(1)

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  ✅ PIPELINE COMPLETE")
print("=" * 65)
print()
print("  Output files:")
outputs = [
    "data/dummy/dummy_inputs.npy",
    "data/dummy/dummy_targets.npy",
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
print("  Handoff to Person 2 (Model Lead):")
print("    ✉️  train_inputs.npy, train_targets.npy, ocean_mask.npy")
print()
print("  Handoff to Person 3 (Viz Lead):")
print("    ✉️  dummy_inputs.npy, dummy_targets.npy, norm_stats.json")
