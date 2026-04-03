"""Schema types shared between dataprism and external consumers (e.g. studio-processor)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ColumnType(StrEnum):
    """Semantic type of a column — how it behaves statistically."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    BINARY = "binary"


class ColumnRole(StrEnum):
    """How the pipeline should treat this column."""

    FEATURE = "feature"
    TARGET = "target"
    IDENTIFIER = "identifier"
    SPLIT = "split"
    VALIDATION = "validation"
    OBSERVATION_DATE = "observation_date"


@dataclass(frozen=True)
class Sentinels:
    """Sentinel codes that a data provider uses instead of null.

    Attributes:
        not_found: Value meaning "record not found in provider"
            (e.g. "-9999"). Indicates a failed lookup — not a true null.
        missing: Value meaning "record found but value unavailable"
            (e.g. "-1"). Represents a real null from the provider.
    """

    not_found: str | None = None
    missing: str | None = None

    def to_dict(self) -> dict[str, str]:
        d: dict[str, str] = {}
        if self.not_found is not None:
            d["not_found"] = self.not_found
        if self.missing is not None:
            d["missing"] = self.missing
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sentinels:
        return cls(
            not_found=data.get("not_found"),
            missing=data.get("missing"),
        )


@dataclass(frozen=True)
class ColumnConfig:
    """Configuration for a single column in the dataset.

    Attributes:
        name: Column name.
        type: Semantic type (continuous, categorical, ordinal, binary).
        role: Pipeline role (feature, target, identifier, split, observation_date).
        description: Human-readable description of this column.
        category: Free-form label for grouping columns
            (e.g. "demographics", "financial", "behavioral"). None when unset.
        provider: Data source / bureau that supplies this column
            (e.g. "experian", "equifax"). None for internal columns.
        sentinels: Provider-specific sentinel codes for no-hit and missing
            values. None when the provider doesn't define sentinels.
    """

    name: str
    type: ColumnType | None = None
    role: ColumnRole = ColumnRole.FEATURE
    description: str | None = None
    category: str | None = None
    provider: str | None = None
    sentinels: Sentinels | None = None

    def is_type(self, *types: ColumnType) -> bool:
        """Check if this column matches any of the given types."""
        return self.type in types

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "role": self.role.value,
        }
        if self.type is not None:
            d["type"] = self.type.value
        if self.description is not None:
            d["description"] = self.description
        if self.category is not None:
            d["category"] = self.category
        if self.provider is not None:
            d["provider"] = self.provider
        if self.sentinels is not None:
            d["sentinels"] = self.sentinels.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ColumnConfig:
        raw_sentinels = data.get("sentinels")
        col_name = data.get("name", "<unknown>")

        raw_role = data.get("role")
        try:
            col_role = ColumnRole(raw_role.lower()) if raw_role is not None else ColumnRole.FEATURE
        except ValueError as err:
            raise ValueError(
                f"Column '{col_name}': invalid role '{raw_role}'. Must be one of: {[r.value for r in ColumnRole]}"
            ) from err

        raw_type = data.get("type")
        try:
            col_type = ColumnType(raw_type.lower()) if raw_type is not None else None
        except ValueError as err:
            raise ValueError(
                f"Column '{col_name}': invalid type '{raw_type}'. Must be one of: {[t.value for t in ColumnType]}"
            ) from err

        return cls(
            name=data["name"],
            role=col_role,
            type=col_type,
            description=data.get("description"),
            category=data.get("category"),
            provider=data.get("provider"),
            sentinels=Sentinels.from_dict(raw_sentinels) if raw_sentinels else None,
        )


class DatasetSchema:
    """Schema for all columns in a dataset.

    Provides O(1) lookup by name and filtered queries by type/role.
    """

    def __init__(self, columns: list[ColumnConfig] | None = None) -> None:
        self._columns: dict[str, ColumnConfig] = {}
        for col in columns or []:
            self._columns[col.name] = col

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._columns

    def __getitem__(self, name: str) -> ColumnConfig:
        return self._columns[name]

    def __len__(self) -> int:
        return len(self._columns)

    def __iter__(self) -> Iterator[str]:
        return iter(self._columns)

    def __repr__(self) -> str:
        roles: dict[str, list[str]] = {}
        for col in self._columns.values():
            roles.setdefault(col.role.value, []).append(col.name)
        parts = ", ".join(f"{role}={len(names)}" for role, names in roles.items())
        return f"DatasetSchema({len(self._columns)} columns: {parts})"

    @property
    def columns(self) -> dict[str, ColumnConfig]:
        """Public read-only view of all column configs keyed by name."""
        return dict(self._columns)

    @property
    def target(self) -> ColumnConfig:
        """Return the single target column. Raises if none or multiple."""
        targets = [col for col in self._columns.values() if col.role is ColumnRole.TARGET]
        if len(targets) != 1:
            raise ValueError(f"Expected exactly 1 target, found {len(targets)}")
        return targets[0]

    # ------------------------------------------------------------------
    # Filtered queries
    # ------------------------------------------------------------------

    def get_features(
        self,
        column_type: ColumnType | None = None,
    ) -> list[ColumnConfig]:
        """Return columns with role=FEATURE, optionally filtered by type."""
        result = [col for col in self._columns.values() if col.role is ColumnRole.FEATURE]
        if column_type is not None:
            result = [col for col in result if col.type is column_type]
        return result

    def get_feature_names(
        self,
        column_type: ColumnType | None = None,
    ) -> list[str]:
        """Shortcut: names of feature columns matching the filter."""
        return [col.name for col in self.get_features(column_type)]

    def get_by_role(self, role: ColumnRole) -> list[ColumnConfig]:
        """Return all columns with the given role."""
        return [col for col in self._columns.values() if col.role is role]

    # ------------------------------------------------------------------
    # Mutation (for derived columns)
    # ------------------------------------------------------------------

    def add(self, config: ColumnConfig) -> None:
        """Add a column config entry. Overwrites if name already exists."""
        self._columns[config.name] = config

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": [col.to_dict() for col in self._columns.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetSchema:
        raw = data.get("columns", [])
        columns = [ColumnConfig.from_dict(c) for c in raw]
        return cls(columns)
