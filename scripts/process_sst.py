"""
Script to process raw CMC SST NetCDF files to the standardized SIH 0.25° grid.

Usage:
    python scripts/process_sst.py --input-dir NASA_sst_data --output-dir data/processed/sst
    python scripts/process_sst.py --input-dir data/raw/sst --output-dir data/processed/sst
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_SST_DIR, PROCESSED_SST_DIR
from src.preprocessing.sst import process_sst_batch


def main():
    parser = argparse.ArgumentParser(description="Process CMC L4 SST to SIH 0.25° grid.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(RAW_SST_DIR if RAW_SST_DIR.exists() else "NASA_sst_data"),
        help="Directory containing raw CMC SST NetCDF files (default: data/raw/sst)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROCESSED_SST_DIR),
        help="Directory to save processed NetCDF files (default: data/processed/sst)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*CMC*.nc",
        help="Glob pattern for raw files (default: *CMC*.nc)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    raw_files = sorted(list(input_dir.glob(args.pattern)))
    if not raw_files:
        print(f"No files matching '{args.pattern}' found in '{input_dir}'.")
        sys.exit(1)

    print(f"Found {len(raw_files)} raw SST files to process.")
    print(f"Output directory: {output_dir}")
    
    saved_files = process_sst_batch(raw_files, output_dir=output_dir)
    
    print("\nProcessing complete!")
    print(f"Successfully saved {len(saved_files)} processed SST files:")
    for f in saved_files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
