# DataPrism Tests

This directory contains test files for the DataPrism package.

## Running Tests

Make sure the package is installed in development mode first:

```bash
pip install -e .
```

Run all tests with pytest:

```bash
python -m pytest tests/
```

Run a specific test file:

```bash
python -m pytest tests/test_default_missing.py
```

## Test Files

- `test_default_missing.py` - Sentinel value handling: continuous/categorical columns with `missing` and `not_found` sentinels, both combined, no-config fallback, integer sentinel conversion
- `test_empty_string_edge_case.py` - Edge cases for empty strings: not replaced when only numeric sentinels configured, explicit empty string sentinel replacement, mixed numeric + empty string sentinels
- `test_feature_filtering.py` - Schema-driven feature selection: `SPLIT` role excluded from analysis but used for stability, `is_target` flag on target variable
- `test_missing_values.py` - Sentinel replacement, provider match rates with schema, fallback match rates from `<provider>_record_not_found` columns
- `test_new_schema.py` - Full EDA run on synthetic data, `DatasetSchema` serialization round-trip with sentinels, `ColumnConfig.is_type()` with single and multiple type args
- `test_real_dataset.py` - Full EDA pipeline with real credit risk CSV + schema: summary structure, top features, data quality, feature detail sections
- `test_stability.py` - Cohort-based stability (train/test split) with real dataset: summary metrics and highest_metrics fields
- `test_time_based_stability.py` - Monthly, weekly, and quartile time window strategies: period structure, sample counts, temporal drift summary
