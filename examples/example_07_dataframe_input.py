"""Example 6: Using DataFrame objects directly with DataPrism."""

import pandas as pd
import numpy as np
from dataprism import (
    DataPrism,
    ColumnConfig, ColumnType, ColumnRole, DatasetSchema,
)


def example_simple_dataframe():
    """Basic usage with a DataFrame object (no schema)."""
    print("\n" + "=" * 70)
    print("Part 1: Basic DataFrame Input")
    print("=" * 70)

    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 20000, 1000),
        'credit_score': np.random.uniform(300, 850, 1000),
        'segment': np.random.choice(['Premium', 'Standard', 'Basic'], 1000),
        'default': np.random.choice([0, 1], 1000, p=[0.8, 0.2])
    })

    prism = DataPrism(max_categories=20)
    results = prism.analyze(data=df, target_variable='default')

    print(f"  Features analyzed: {len(results['features'])}")
    print(f"  Target: {results['metadata']['target_variable']}")


def example_with_schema():
    """Using DataFrame with a programmatic DatasetSchema."""
    print("\n" + "=" * 70)
    print("Part 2: DataFrame + DatasetSchema")
    print("=" * 70)

    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, 1000),
        'annual_income': np.random.normal(50000, 20000, 1000),
        'credit_score': np.random.uniform(300, 850, 1000),
        'employment_status': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], 1000),
        'loan_approved': np.random.choice([0, 1], 1000)
    })

    schema = DatasetSchema([
        ColumnConfig('age', ColumnType.CONTINUOUS, ColumnRole.FEATURE,
                     provider='internal', description='Customer age in years'),
        ColumnConfig('annual_income', ColumnType.CONTINUOUS, ColumnRole.FEATURE,
                     provider='credit_bureau', description='Annual income in USD'),
        ColumnConfig('credit_score', ColumnType.CONTINUOUS, ColumnRole.FEATURE,
                     provider='credit_bureau', description='FICO credit score'),
        ColumnConfig('employment_status', ColumnType.CATEGORICAL, ColumnRole.FEATURE,
                     provider='internal', description='Current employment status'),
        ColumnConfig('loan_approved', ColumnType.BINARY, ColumnRole.TARGET),
    ])

    prism = DataPrism(max_categories=20)
    results = prism.analyze(data=df, schema=schema, target_variable='loan_approved')

    print(f"  Features analyzed: {len(results['features'])}")
    print(f"  Schema available: {results['metadata']['schema_available']}")

    if 'provider_match_rates' in results['summary']:
        for provider, stats in results['summary']['provider_match_rates'].items():
            print(f"  Provider {provider}: {stats['match_rate']:.1%}")


if __name__ == "__main__":
    print("=" * 70)
    print("DataPrism - DataFrame Input Examples")
    print("=" * 70)

    example_simple_dataframe()
    example_with_schema()

    print("\n" + "=" * 70)
    print("Key takeaways:")
    print("  1. DataPrism.analyze() accepts DataFrame objects directly")
    print("  2. Use DataLoader utilities to load CSV/Parquet files")
    print("  3. Use DatasetSchema + ColumnConfig for column configuration")
    print("=" * 70)
