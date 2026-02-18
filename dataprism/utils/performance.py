"""Performance optimization utilities for EDA processing."""

import hashlib
import json
import pickle
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from dataprism.utils.logger import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Manages caching of expensive computations."""

    def __init__(self, cache_dir: Optional[Path] = None, enabled: bool = True):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache files (default: .eda_cache)
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.cache_dir = cache_dir or Path(".eda_cache")
        if self.enabled:
            self.cache_dir.mkdir(exist_ok=True)
            logger.debug("Cache initialized at %s", self.cache_dir)

    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """
        Generate a unique cache key for function call.

        Args:
            func_name: Name of the function
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Cache key hash
        """
        # Create a hashable representation
        key_data = {
            "func": func_name,
            "args": str(args),
            "kwargs": str(sorted(kwargs.items())),
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if not self.enabled:
            return None

        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    logger.debug("Cache hit: %s", key)
                    return pickle.load(f)
            except Exception as e:
                logger.warning("Failed to load cache %s: %s", key, e)
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.enabled:
            return

        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(value, f)
            logger.debug("Cache stored: %s", key)
        except Exception as e:
            logger.warning("Failed to save cache %s: %s", key, e)

    def clear(self) -> None:
        """Clear all cached files."""
        if not self.enabled or not self.cache_dir.exists():
            return

        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning("Failed to delete cache file %s: %s", cache_file, e)

        logger.info("Cache cleared")


def cached(cache_manager: Optional[CacheManager] = None) -> Callable:
    """
    Decorator to cache function results.

    Args:
        cache_manager: CacheManager instance to use

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if cache_manager is None or not cache_manager.enabled:
                return func(*args, **kwargs)

            # Generate cache key
            cache_key = cache_manager._generate_cache_key(func.__name__, args, kwargs)

            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result)
            return result

        return wrapper

    return decorator


class ChunkedCSVReader:
    """Reads large CSV files in chunks for memory-efficient processing."""

    def __init__(
        self,
        filepath: Path,
        chunksize: int = 10000,
        sample_size: Optional[int] = None,
    ):
        """
        Initialize chunked CSV reader.

        Args:
            filepath: Path to CSV file
            chunksize: Number of rows per chunk
            sample_size: Optional max rows to read (for sampling)
        """
        self.filepath = filepath
        self.chunksize = chunksize
        self.sample_size = sample_size

    def read_all(self) -> pd.DataFrame:
        """
        Read entire file (with optional sampling) using chunks.

        Returns:
            Complete DataFrame
        """
        chunks = []
        rows_read = 0

        for chunk in pd.read_csv(self.filepath, chunksize=self.chunksize):
            chunks.append(chunk)
            rows_read += len(chunk)

            if self.sample_size and rows_read >= self.sample_size:
                # Truncate last chunk if needed
                if rows_read > self.sample_size:
                    excess = rows_read - self.sample_size
                    chunks[-1] = chunks[-1].iloc[:-excess]
                break

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks, ignore_index=True)
        logger.info("Loaded %s rows from %s", len(df), self.filepath.name)
        return df

    def process_chunks(self, processor: Callable[[pd.DataFrame], Any]) -> list:
        """
        Process file in chunks without loading all into memory.

        Args:
            processor: Function to process each chunk

        Returns:
            List of results from each chunk
        """
        results = []
        rows_processed = 0

        for i, chunk in enumerate(pd.read_csv(self.filepath, chunksize=self.chunksize)):
            logger.debug("Processing chunk %s (%s rows)", i + 1, len(chunk))
            result = processor(chunk)
            results.append(result)

            rows_processed += len(chunk)
            if self.sample_size and rows_processed >= self.sample_size:
                break

        logger.info("Processed %s rows in %s chunks", rows_processed, len(results))
        return results


def should_use_chunking(filepath: Path, threshold_mb: float = 100.0) -> bool:
    """
    Determine if chunked reading should be used based on file size.

    Args:
        filepath: Path to CSV file
        threshold_mb: File size threshold in MB

    Returns:
        True if chunking recommended
    """
    file_size_mb = filepath.stat().st_size / (1024 * 1024)
    return file_size_mb > threshold_mb
