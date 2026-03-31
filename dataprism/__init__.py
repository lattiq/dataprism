"""DataPrism - Lightweight EDA library for data analysis."""

from importlib.metadata import PackageNotFoundError, version

from dataprism.analyzers.woe import WoEAnalyzer, WoEResult
from dataprism.data.loader import DataLoader
from dataprism.eda import DataPrism
from dataprism.exceptions import (
    AnalysisError,
    ConfigurationError,
    DataLoadError,
    DataPrismError,
    DataValidationError,
    FeatureTypeError,
    MissingDataError,
    OutputFormattingError,
    StabilityAnalysisError,
    TargetAnalysisError,
)
from dataprism.explorer import Explorer
from dataprism.schema import (
    ColumnConfig,
    ColumnRole,
    ColumnType,
    DatasetSchema,
    Sentinels,
)

try:
    __version__ = version("dataprism")
except PackageNotFoundError:
    __version__ = "0.1.4"

__author__ = "LattIQ Development Team"
__email__ = "dev@lattiq.com"

__all__ = [
    "DataPrism",
    # Data Loading Utilities
    "DataLoader",
    # Schema types
    "ColumnConfig",
    "ColumnType",
    "ColumnRole",
    "Sentinels",
    "DatasetSchema",
    # WoE / IV
    "WoEAnalyzer",
    "WoEResult",
    # Explorer
    "Explorer",
    # Exceptions
    "DataPrismError",
    "DataLoadError",
    "DataValidationError",
    "AnalysisError",
    "ConfigurationError",
    "FeatureTypeError",
    "MissingDataError",
    "StabilityAnalysisError",
    "OutputFormattingError",
    "TargetAnalysisError",
]
