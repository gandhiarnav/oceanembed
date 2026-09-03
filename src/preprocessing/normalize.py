"""
OceanEmbed — Normalization & Tensor Export Pipeline.
Performs z-score normalization on all 7 input channels strictly on ocean pixels,
isolates statistics to the training partition to prevent data leakage,
and exports final ML tensors (.npy), ocean mask, and normalization statistics (norm_stats.json).
"""
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
import xarray as xr

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    CANONICAL_CHANNELS,
    TARGET_LAT,
    TARGET_LON,
    TARGET_DEPTHS,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    DEFAULT_TEST_RATIO,
)


def compute_ocean_mask(ds_tgt: xr.Dataset, ds_surf: xr.Dataset) -> np.ndarray:
    """
    Construct the physical master ocean mask based on the target domain.
    A pixel is ocean if it is non-NaN in the surface target temperature.
    """
    tgt_var = "thetao" if "thetao" in ds_tgt else list(ds_tgt.data_vars)[0]
    # Check surface layer (depth=0) of target
    if "depth" in ds_tgt.dims:
        sample_tgt = ds_tgt[tgt_var].isel(time=0).sel(depth=0, method="nearest").values
    else:
        sample_tgt = ds_tgt[tgt_var].isel(time=0, deptht=0).values

    ocean_mask = ~np.isnan(sample_tgt)
    return ocean_mask


def normalize_and_export_tensors(
    processed_dir: Path = PROCESSED_DATA_DIR,
    output_dir: Optional[Path] = None,
    split: bool = False,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
) -> Dict[str, np.ndarray]:
    """
    Load regridded surface inputs and target datasets, partition temporally (if split=True),
    compute z-score normalization statistics exclusively on the training partition,
    and export tensors, ocean mask, norm_stats.json, and split_info.json.
    """
    out_dir = Path(output_dir) if output_dir else processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  OceanEmbed — Normalization & Tensor Export (7-Channel Pipeline)")
    print("=" * 65)

    surface_path = processed_dir / "surface_inputs_regridded.nc"
    target_path = processed_dir / "target_temp_regridded.nc"

    if not surface_path.exists():
        raise FileNotFoundError(f"Surface file not found: {surface_path}. Run regrid_harmonize first.")
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}. Run regrid_harmonize first.")

    ds_surf = xr.open_dataset(surface_path)
    ds_tgt = xr.open_dataset(target_path)
    print(f"  Loaded surface inputs: {surface_path}")
    print(f"  Loaded target dataset: {target_path}")

    # Build Master Ocean Mask from physical target domain
    ocean_mask = compute_ocean_mask(ds_tgt, ds_surf)
    mask_path = out_dir / "ocean_mask.npy"
    np.save(mask_path, ocean_mask)
    print(f"  Ocean mask saved: {mask_path}")
    print(f"    Ocean pixels: {ocean_mask.sum()} / {ocean_mask.size} ({ocean_mask.mean() * 100:.1f}%)\n")

    # Map Available Variables to Canonical Channels
    active_channels = []
    for canon_name, aliases in CANONICAL_CHANNELS:
        found_var = None
        for alias in aliases:
            if alias in ds_surf.data_vars:
                found_var = alias
                break
        if found_var:
            active_channels.append((canon_name, found_var))
        else:
            print(f"  ⚠️  Channel '{canon_name}' not found in dataset. (Aliases searched: {aliases})")

    print(f"  Active channels for export ({len(active_channels)}/{len(CANONICAL_CHANNELS)}):")
    for idx, (cname, vname) in enumerate(active_channels):
        print(f"    Ch {idx}: {cname:8s} (from '{vname}')")
    print()

    if len(active_channels) != len(CANONICAL_CHANNELS):
        missing = [c for c, _ in CANONICAL_CHANNELS if c not in [a for a, _ in active_channels]]
        raise ValueError(f"Missing channels: {missing}. Contract requires all {len(CANONICAL_CHANNELS)} channels.")

    n_times = len(ds_surf.time)
    n_lat = len(TARGET_LAT)
    n_lon = len(TARGET_LON)
    n_chan = len(active_channels)

    raw_inputs = np.zeros((n_times, n_chan, n_lat, n_lon), dtype=np.float32)
    for ch_idx, (canon_name, var_name) in enumerate(active_channels):
        raw_inputs[:, ch_idx, :, :] = ds_surf[var_name].values.astype(np.float32)

    tgt_var_name = "thetao" if "thetao" in ds_tgt else list(ds_tgt.data_vars)[0]
    raw_targets = ds_tgt[tgt_var_name].values.astype(np.float32)

    time_values = [str(t)[:10] for t in ds_surf.time.values]

    # Partition into Splits (Temporal Train / Val / Test)
    splits = {}
    if split and n_times >= 3:
        n_train = max(1, int(round(n_times * train_ratio)))
        n_val = max(1, int(round(n_times * val_ratio))) if n_times >= 5 else 1
        n_test = n_times - (n_train + n_val)
        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1

        splits["train"] = {
            "indices": slice(0, n_train),
            "dates": time_values[0:n_train],
        }
        splits["val"] = {
            "indices": slice(n_train, n_train + n_val),
            "dates": time_values[n_train:n_train + n_val],
        }
        splits["test"] = {
            "indices": slice(n_train + n_val, n_times),
            "dates": time_values[n_train + n_val:n_times],
        }
        print(f"  Splits: Train = {n_train} days, Val = {n_val} days, Test = {n_test} days")
    else:
        splits["train"] = {
            "indices": slice(0, n_times),
            "dates": time_values,
        }
        print(f"  Single partition export: Train = {n_times} days")

    # Compute Normalization Statistics EXCLUSIVELY on Training Split
    print("\n  Computing normalization statistics (strictly on training split)...")
    train_idx = splits["train"]["indices"]
    train_raw_inputs = raw_inputs[train_idx]  # (N_train, 7, 101, 241)

    norm_stats = {}
    for ch_idx, (canon_name, var_name) in enumerate(active_channels):
        train_channel = train_raw_inputs[:, ch_idx, :, :]
        ocean_vals = train_channel[:, ocean_mask]

        mean_val = float(np.nanmean(ocean_vals))
        std_val = float(np.nanstd(ocean_vals))
        if std_val < 1e-8 or np.isnan(std_val):
            std_val = 1.0

        norm_stats[canon_name] = {"mean": mean_val, "std": std_val}
        print(f"    {canon_name:8s}: mean = {mean_val:8.4f}, std = {std_val:8.4f}")

    # Save norm_stats.json
    stats_path = out_dir / "norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"\n  Saved normalization stats: {stats_path}")

    # Apply Normalization and Export Tensors for Each Split
    results = {}
    split_metadata = {}

    for split_name, sinfo in splits.items():
        s_idx = sinfo["indices"]
        s_inputs = raw_inputs[s_idx].copy()
        s_targets = raw_targets[s_idx].copy()

        # Normalize inputs using training statistics
        for ch_idx, (canon_name, _) in enumerate(active_channels):
            mean_val = norm_stats[canon_name]["mean"]
            std_val = norm_stats[canon_name]["std"]

            s_inputs[:, ch_idx, :, :] = (s_inputs[:, ch_idx, :, :] - mean_val) / std_val
            # Land pixels and residual coastal NaNs filled with 0.0 (neutral mean)
            s_inputs[:, ch_idx, ~ocean_mask] = 0.0
            s_inputs[:, ch_idx, :, :] = np.nan_to_num(s_inputs[:, ch_idx, :, :], nan=0.0)

        # Fill land NaNs in targets with 0.0
        s_targets = np.nan_to_num(s_targets, nan=0.0)

        in_path = out_dir / f"{split_name}_inputs.npy"
        tgt_path = out_dir / f"{split_name}_targets.npy"

        np.save(in_path, s_inputs)
        np.save(tgt_path, s_targets)

        results[f"{split_name}_inputs"] = s_inputs
        results[f"{split_name}_targets"] = s_targets

        split_metadata[split_name] = {
            "num_timesteps": int(s_inputs.shape[0]),
            "date_range": [sinfo["dates"][0], sinfo["dates"][-1]] if sinfo["dates"] else [],
            "input_shape": list(s_inputs.shape),
            "target_shape": list(s_targets.shape),
        }

        print(f"  Saved {split_name:5s} inputs : {in_path} shape={s_inputs.shape} dtype={s_inputs.dtype}")
        print(f"  Saved {split_name:5s} targets: {tgt_path} shape={s_targets.shape} dtype={s_targets.dtype}")

    # Save split_info.json
    split_info_path = out_dir / "split_info.json"
    with open(split_info_path, "w") as f:
        json.dump(split_metadata, f, indent=2)
    print(f"\n  Saved split info: {split_info_path}")

    # Verification Checks
    print("\n" + "=" * 65)
    print("  FORMAT & CONTRACT VERIFICATION")
    print("=" * 65)
    for split_name in splits:
        in_t = results[f"{split_name}_inputs"]
        tgt_t = results[f"{split_name}_targets"]
        assert in_t.dtype == np.float32, f"Inputs dtype mismatch"
        assert tgt_t.dtype == np.float32, f"Targets dtype mismatch"
        assert np.isnan(in_t).sum() == 0, f"Inputs contain NaNs"
        assert np.isnan(tgt_t).sum() == 0, f"Targets contain NaNs"
        print(f"  ✅ {split_name:5s} inputs shape: {in_t.shape}, Targets shape: {tgt_t.shape}, NaNs: 0")

    print("=" * 65)
    print("  ✅ ALL CHECKS PASSED — Leakage-free tensor export complete!")
    print("=" * 65)

    return results


def main():
    parser = argparse.ArgumentParser(description="Normalize and export ML tensors.")
    parser.add_argument("--split", action="store_true", help="Split data temporally into train/val/test.")
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Train ratio.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Test ratio.")

    args = parser.parse_args()
    normalize_and_export_tensors(
        split=args.split,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )


if __name__ == "__main__":
    main()
