"""Result types used by analyzers."""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DatasetInfo:
    """Basic dataset information."""
    rows: int
    columns: int
    memory_mb: float
    missing_cells: int
    missing_percentage: float
    duplicate_rows: int = 0


@dataclass
class MissingInfo:
    """Missing data information."""
    count: int
    percent: float


@dataclass
class ContinuousStats:
    """Statistics for continuous features."""
    count: int
    mean: float
    std: float
    min: float
    max: float
    q1: float
    median: float
    q3: float
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None


@dataclass
class CategoricalStats:
    """Statistics for categorical features."""
    count: int
    unique: int
    mode: Any
    mode_count: int
    value_counts: Dict[str, int]
    value_percentages: Dict[str, float]
