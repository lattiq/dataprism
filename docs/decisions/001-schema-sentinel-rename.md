# ADR-001: Rename sentinel fields from default/no_hit_value to missing/not_found

## Status
Accepted

## Context

The legacy schema (`core/types.py`) used `FeatureMetadata` with flat fields:
- `default: Optional[str]` — value when record found but data unavailable
- `no_hit_value: Optional[str]` — value when record not found in provider

The name `default` was ambiguous — it could be confused with a default/fallback value rather than a sentinel for missing data.

## Decision

Introduced a `Sentinels` dataclass in `schema.py` with clearer naming:
- `missing: Optional[str]` — replaces `default`
- `not_found: Optional[str]` — replaces `no_hit_value`

Sentinels are nested inside `ColumnConfig` rather than being flat fields, grouping provider-specific sentinel codes together.

## Consequences

- Clearer semantics: `missing` explicitly signals "provider returned a sentinel for missing data"
- Better encapsulation: sentinel codes grouped in their own dataclass
- Schema JSON uses `"sentinels": {"not_found": "...", "missing": "..."}` instead of top-level `"default"` / `"no_hit_value"`
- Legacy format compatibility handled via `output/mapper.py` for downstream consumers
