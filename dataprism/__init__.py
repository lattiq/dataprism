"""DataPrism - Lightweight EDA library for data analysis."""

from importlib.metadata import version, PackageNotFoundError

from dataprism.exceptions import (
    AnalysisError,
    ConfigurationError,
    DataLoadError,
    DataValidationError,
    DataPrismError,
    FeatureTypeError,
    MissingDataError,
    OutputFormattingError,
    StabilityAnalysisError,
    TargetAnalysisError,
)
from dataprism.data.loader import DataLoader
from dataprism.schema import (
    ColumnConfig,
    ColumnRole,
    ColumnType,
    DatasetSchema,
    Sentinels,
)
from dataprism.eda import EDARunner
from dataprism.viewer import serve_results

try:
    __version__ = version("dataprism")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "LattIQ Development Team"
__email__ = "dev@lattiq.com"

__all__ = [
    "EDARunner",
    # Viewer
    "serve_results",
    # Data Loading Utilities
    "DataLoader",
    # Schema types
    "ColumnConfig",
    "ColumnType",
    "ColumnRole",
    "Sentinels",
    "DatasetSchema",
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
