"""Convert CSV files to Parquet format for faster loading."""

import argparse
import sys
from pathlib import Path

import pandas as pd


def convert_csv_to_parquet(
    csv_path: str,
    parquet_path: str = None,
    compression: str = "snappy",
    show_stats: bool = True
) -> None:
    """
    Convert a CSV file to Parquet format.

    Args:
        csv_path: Path to input CSV file
        parquet_path: Path to output Parquet file (optional, auto-generated if not provided)
        compression: Compression algorithm ('snappy', 'gzip', 'brotli', 'lz4', 'zstd', 'none')
        show_stats: Whether to print conversion statistics
    """
    csv_path = Path(csv_path)

    # Validate input file exists
    if not csv_path.exists():
        print(f"❌ Error: CSV file not found: {csv_path}")
        sys.exit(1)

    # Auto-generate output path if not provided
    if parquet_path is None:
        parquet_path = csv_path.with_suffix('.parquet')
    else:
        parquet_path = Path(parquet_path)

    # Create output directory if needed
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # Get input file size
    csv_size_mb = csv_path.stat().st_size / 1024 / 1024

    print(f"Converting: {csv_path}")
    print(f"       to: {parquet_path}")
    print(f"Compression: {compression}")
    print()

    # Load CSV
    print(f"Loading CSV ({csv_size_mb:.1f} MB)...")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

    # Save as Parquet
    print(f"\nConverting to Parquet...")
    df.to_parquet(
        parquet_path,
        engine='pyarrow',
        compression=compression,
        index=False
    )

    # Get output file size
    parquet_size_mb = parquet_path.stat().st_size / 1024 / 1024

    # Show statistics
    if show_stats:
        size_ratio = (1 - parquet_size_mb / csv_size_mb) * 100 if csv_size_mb > 0 else 0
        print(f"✅ Conversion complete!")
        print()
        print("=" * 60)
        print("Statistics:")
        print("=" * 60)
        print(f"  CSV size:     {csv_size_mb:>10.2f} MB")
        print(f"  Parquet size: {parquet_size_mb:>10.2f} MB")
        print(f"  Reduction:    {size_ratio:>10.1f}%")
        print(f"  Ratio:        {parquet_size_mb/csv_size_mb if csv_size_mb > 0 else 0:>10.2f}x smaller")
        print("=" * 60)


def main():
    """Main entry point for the conversion script."""
    parser = argparse.ArgumentParser(
        description="Convert CSV files to Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a single file (auto-generate output name)
  python convert_csv_to_parquet.py data.csv

  # Specify output file
  python convert_csv_to_parquet.py data.csv output.parquet

  # Use different compression
  python convert_csv_to_parquet.py data.csv --compression gzip

  # Convert tmp/dataset.csv to tmp/dataset.parquet
  python convert_csv_to_parquet.py tmp/dataset.csv

Compression options:
  - snappy: Fast compression (default, recommended)
  - gzip:   Better compression ratio, slower
  - brotli: Best compression ratio, slowest
  - lz4:    Very fast, lower compression
  - zstd:   Balanced speed and compression
  - none:   No compression
        """
    )

    parser.add_argument(
        "csv_file",
        help="Path to CSV file to convert"
    )
    parser.add_argument(
        "parquet_file",
        nargs="?",
        default=None,
        help="Path to output Parquet file (optional, auto-generated if not provided)"
    )
    parser.add_argument(
        "--compression",
        "-c",
        choices=["snappy", "gzip", "brotli", "lz4", "zstd", "none"],
        default="snappy",
        help="Compression algorithm (default: snappy)"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Don't show conversion statistics"
    )

    args = parser.parse_args()

    try:
        convert_csv_to_parquet(
            csv_path=args.csv_file,
            parquet_path=args.parquet_file,
            compression=args.compression,
            show_stats=not args.quiet
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
