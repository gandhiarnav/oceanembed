"""
Verification script for SIH-26066 dataset contracts (dummy or processed real data).

Usage:
    python scripts/verify_data.py
    python scripts/verify_data.py --dir dummy_data/dummy
    python scripts/verify_data.py --dir data/processed --prefix train
"""
import sys
import argparse
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DUMMY_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHANNEL_NAMES,
    TARGET_DEPTHS,
    NUM_LAT,
    NUM_LON,
)


def verify_dataset(data_dir: Path, prefix: str = "dummy"):
    print("=" * 60)
    print(f"       OCEANEMBED DATASET VERIFICATION ({data_dir})")
    print("=" * 60)

    inputs_file = data_dir / f"{prefix}_inputs.npy"
    targets_file = data_dir / f"{prefix}_targets.npy"
    dates_file = data_dir / f"{prefix}_dates.npy"

    if not inputs_file.exists():
        print(f"❌ Inputs file not found: {inputs_file}")
        return False

    inputs = np.load(inputs_file)
    targets = np.load(targets_file) if targets_file.exists() else None
    dates = np.load(dates_file, allow_pickle=True) if dates_file.exists() else None

    n_samples = inputs.shape[0]

    # 1. SHAPES & DTYPES
    print("\n[1] SHAPES & DTYPES:")
    print(f"  {inputs_file.name}  : shape={inputs.shape}, dtype={inputs.dtype}")
    if targets is not None:
        print(f"  {targets_file.name} : shape={targets.shape}, dtype={targets.dtype}")
    if dates is not None:
        print(f"  {dates_file.name}   : shape={dates.shape}")

    expected_inputs = (n_samples, 7, NUM_LAT, NUM_LON)
    expected_targets = (n_samples, len(TARGET_DEPTHS), NUM_LAT, NUM_LON)

    inputs_shape_ok = inputs.shape == expected_inputs
    inputs_dtype_ok = inputs.dtype == np.float32

    print(f"  inputs shape OK  : {inputs_shape_ok}  (expected {expected_inputs})")
    print(f"  inputs dtype OK  : {inputs_dtype_ok}")

    targets_shape_ok = True
    targets_dtype_ok = True
    if targets is not None:
        targets_shape_ok = targets.shape == expected_targets
        targets_dtype_ok = targets.dtype == np.float32
        print(f"  targets shape OK : {targets_shape_ok}  (expected {expected_targets})")
        print(f"  targets dtype OK : {targets_dtype_ok}")

    # 2. SPATIAL GRID
    print("\n[2] SPATIAL GRID:")
    print(f"  Lat points : {inputs.shape[2]}  (expected {NUM_LAT} -> 5°N to 30°N at 0.25°)")
    print(f"  Lon points : {inputs.shape[3]}  (expected {NUM_LON} -> 45°E to 105°E at 0.25°)")

    # 3. TEMPORAL
    if dates is not None:
        print("\n[3] TEMPORAL:")
        print(f"  Days       : {len(dates)}")
        print(f"  Date range : {dates[0]} to {dates[-1]}")

    # 4. NaN CHECK
    nan_inputs = int(np.isnan(inputs).sum())
    print("\n[4] NaN CHECK:")
    print(f"  NaNs in inputs  : {nan_inputs}")
    if targets is not None:
        nan_targets = int(np.isnan(targets).sum())
        print(f"  NaNs in targets : {nan_targets}")

    # 5. INPUT CHANNEL SUMMARY
    print("\n[5] INPUT CHANNELS:")
    print(f"  {'Idx':<5}{'Channel':<12}{'Min':>10}{'Max':>10}{'Mean':>10}{'Std':>10}")
    print("  " + "-" * 55)
    for i, name in enumerate(CHANNEL_NAMES):
        if i < inputs.shape[1]:
            ch = inputs[:, i, :, :]
            print(f"  {i:<5}{name:<12}{ch.min():>10.4f}{ch.max():>10.4f}{ch.mean():>10.4f}{ch.std():>10.4f}")

    # 6. TARGET DEPTH LEVELS
    if targets is not None:
        print("\n[6] TARGET DEPTH TEMPERATURES:")
        print(f"  {'Idx':<5}{'Depth_m':<10}{'Min':>10}{'Max':>10}{'Mean':>10}{'Std':>10}")
        print("  " + "-" * 55)
        for i, d in enumerate(TARGET_DEPTHS):
            if i < targets.shape[1]:
                lv = targets[:, i, :, :]
                print(f"  {i:<5}{d:<10}{lv.min():>10.4f}{lv.max():>10.4f}{lv.mean():>10.4f}{lv.std():>10.4f}")

    print("\n" + "=" * 60)
    all_ok = (
        inputs_shape_ok
        and inputs_dtype_ok
        and targets_shape_ok
        and targets_dtype_ok
        and nan_inputs == 0
        and (nan_targets == 0 if targets is not None else True)
    )
    status = "OK" if all_ok else "FAIL"
    print(f"  [{status}] Data matches expected SIH-26066 format: {all_ok}")
    print("=" * 60)
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Verify Ocean ML dataset tensors.")
    parser.add_argument(
        "--dir",
        type=str,
        default=str(DUMMY_DATA_DIR),
        help="Directory containing numpy tensors (default: dummy_data/dummy)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="dummy",
        help="Filename prefix for inputs/targets (default: 'dummy', or 'train')",
    )

    args = parser.parse_args()
    data_dir = Path(args.dir)
    if not data_dir.exists():
        # Fallback check
        alt = PROJECT_ROOT / "data" / "dummy"
        if alt.exists():
            data_dir = alt

    verify_dataset(data_dir, prefix=args.prefix)


if __name__ == "__main__":
    main()
