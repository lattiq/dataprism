"""Data loading and validation utilities."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from dataprism.schema import DatasetSchema
from dataprism.utils.logger import get_logger
from dataprism.utils.performance import ChunkedCSVReader, should_use_chunking

logger = get_logger(__name__)


class DataLoader:
    """Handles loading and validation of datasets."""

    @staticmethod
    def load_csv(
        filepath: Union[str, Path],
        sample_size: Optional[int] = None,
        use_chunking: Optional[bool] = None,
        chunk_size: int = 50000,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load CSV file with optional sampling and chunking for large files.

        Args:
            filepath: Path to CSV file
            sample_size: Optional number of rows to sample
            use_chunking: Force chunking on/off. If None, auto-detect based on file size
            chunk_size: Number of rows per chunk when using chunking
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        # Auto-detect chunking if not specified
        if use_chunking is None:
            use_chunking = should_use_chunking(filepath, threshold_mb=100.0)
            if use_chunking:
                logger.info("Large file detected (%.1f MB), using chunked reading", filepath.stat().st_size / 1024 / 1024)

        # Use chunked reading for large files
        try:
            if use_chunking:
                reader = ChunkedCSVReader(filepath, chunksize=chunk_size, sample_size=sample_size)
                df = reader.read_all()
            else:
                if sample_size:
                    df = pd.read_csv(filepath, nrows=sample_size, **kwargs)
                else:
                    df = pd.read_csv(filepath, **kwargs)
        except pd.errors.ParserError as e:
            raise ValueError(
                f"CSV file appears malformed: {filepath}. "
                f"Check for inconsistent delimiters or quoting. Details: {e}"
            ) from e
        except UnicodeDecodeError as e:
            raise ValueError(
                f"CSV file has encoding issues: {filepath}. "
                f"Try specifying encoding='utf-8' or 'latin-1'. Details: {e}"
            ) from e

        return df

    @staticmethod
    def load_parquet(
        filepath: Union[str, Path],
        sample_size: Optional[int] = None,
        columns: Optional[list] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load Parquet file with optional column selection and sampling.

        Args:
            filepath: Path to Parquet file
            sample_size: Optional number of rows to sample
            columns: Optional list of columns to load (reduces memory usage)
            **kwargs: Additional arguments for pd.read_parquet

        Returns:
            Loaded DataFrame
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Parquet file not found: {filepath}")

        # Log file size
        file_size_mb = filepath.stat().st_size / 1024 / 1024
        logger.info("Loading Parquet file (%.1f MB)...", file_size_mb)

        # Read parquet file
        try:
            df = pd.read_parquet(
                filepath,
                columns=columns,
                engine='pyarrow',
                **kwargs
            )

            # Apply sampling if requested
            if sample_size and len(df) > sample_size:
                logger.info("Sampling %s rows from %s total rows", sample_size, len(df))
                df = df.head(sample_size)

            logger.info("Loaded %s rows and %s columns", len(df), len(df.columns))
            return df

        except ImportError as e:
            raise ImportError(
                "Parquet support requires 'pyarrow'. "
                "Install it with: pip install pyarrow"
            ) from e

    @staticmethod
    def load_schema(
        filepath: Union[str, Path]
    ) -> DatasetSchema:
        """
        Load dataset schema from JSON file.

        Args:
            filepath: Path to JSON file with schema (expects {"columns": [...]})

        Returns:
            DatasetSchema instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning("Schema file not found: %s. Proceeding with auto-detected types.", filepath)
            return DatasetSchema()

        try:
            with open(filepath) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Schema file contains invalid JSON: {filepath}. Details: {e}") from e

        return DatasetSchema.from_dict(data)

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate DataFrame and return basic information.

        Args:
            df: DataFrame to validate

        Returns:
            Dictionary with validation results
        """
        if df.empty:
            raise ValueError("DataFrame is empty")

        return {
            "rows": len(df),
            "columns": len(df.columns),
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "column_types": df.dtypes.to_dict(),
            "has_duplicates": df.duplicated().any(),
            "duplicate_count": df.duplicated().sum()
        }
