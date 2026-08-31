"""
OceanEmbed — End-to-End Multi-Source Data Pipeline Runner.
Coordinates data downloads, spatial harmonization/regridding, and tensor normalization.

Usage:
    python scripts/run_pipeline.py --skip-download
    python scripts/run_pipeline.py --start 2020-01-01 --end 2020-01-31
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DUMMY_DATA_DIR,
)


def run_step(step_name: str, script_path: Path, optional: bool = False, dry_run: bool = False) -> bool:
    """Run a pipeline step and return True on success."""
    print(f"\n{'─' * 65}")
    print(f"  STEP: {step_name}")
    print(f"{'─' * 65}")
    if dry_run:
        print(f"  [DRY RUN] Would run: python {script_path}")
        return True

    if not script_path.exists():
        print(f"  ⚠️  Script not found: {script_path}")
        return optional

    result = subprocess.run([sys.executable, str(script_path)], capture_output=False)
    if result.returncode != 0:
        if optional:
            print(f"\n  ⚠️  Optional step failed or skipped: {step_name}")
            return True
        else:
            print(f"\n  ❌ Step failed: {step_name}")
            return False

    print(f"\n  ✅ Step complete: {step_name}")
    return True


def main():
    parser = argparse.ArgumentParser(description="OceanEmbed — End-to-End Data Pipeline")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2020-01-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--skip-download", action="store_true", help="Skip download step (use existing raw files)")
    parser.add_argument("--dry-run", action="store_true", help="Print steps without executing them")

    args = parser.parse_args()

    # Validate dates
    try:
        start_dt = datetime.strptime(args.start, "%Y-%m-%d")
        end_dt = datetime.strptime(args.end, "%Y-%m-%d")
        assert start_dt <= end_dt, "Start date must be before end date"
    except (ValueError, AssertionError) as e:
        print(f"  ❌ Invalid date range: {e}")
        sys.exit(1)

    print("=" * 65)
    print("  OceanEmbed — End-to-End Multi-Source Data Pipeline")
    print("=" * 65)
    print(f"  Date range : {args.start} → {args.end}")
    print(f"  Mode       : {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    pipeline_steps = []

    # Step 1: Downloads (if not skipped)
    if not args.skip_download:
        pipeline_steps.append((
            "Download CMEMS GLORYS 3D Target Temperature",
            PROJECT_ROOT / "src" / "download" / "download_glorys.py",
            False
        ))
        pipeline_steps.append((
            "Download CMEMS GLORYS Surface Inputs (SST, SSH, U/V Currents)",
            PROJECT_ROOT / "src" / "download" / "download_surface.py",
            False
        ))
        pipeline_steps.append((
            "Download NASA SMAP Sea Surface Salinity (SSS)",
            PROJECT_ROOT / "src" / "download" / "download_sss.py",
            True
        ))
        pipeline_steps.append((
            "Download ECMWF ERA5 Surface Winds (10m U/V Winds)",
            PROJECT_ROOT / "src" / "download" / "download_era5.py",
            True
        ))
    else:
        print("  ⏭️  Skipping download steps (--skip-download)")

    # Step 2: SST Batch Processing (if raw CMC files present)
    raw_sst_files = list(RAW_DATA_DIR.glob("sst/*CMC*.nc"))
    if raw_sst_files:
        pipeline_steps.append((
            "Process CMC Satellite SST to 0.25° Grid",
            PROJECT_ROOT / "scripts" / "process_sst.py",
            True
        ))

    # Step 3: Spatial Harmonization & Multi-Source Regridding
    pipeline_steps.append((
        "Spatial Harmonization & Multi-Source Regridding (0.25° Grid)",
        PROJECT_ROOT / "src" / "preprocessing" / "regrid_harmonize.py",
        False
    ))

    # Step 4: Normalization & Final Tensor Export (if full 7 channels available)
    pipeline_steps.append((
        "Z-Score Normalization & Tensor Export",
        PROJECT_ROOT / "src" / "preprocessing" / "normalize.py",
        True
    ))

    total = len(pipeline_steps)
    for i, (step_name, script_path, optional) in enumerate(pipeline_steps, 1):
        print(f"\n[{i}/{total}] {step_name}")
        success = run_step(step_name, script_path, optional=optional, dry_run=args.dry_run)
        if not success:
            print(f"\n  ❌ Pipeline aborted at step {i}/{total}: {step_name}")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("  ✅ PIPELINE RUN COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
