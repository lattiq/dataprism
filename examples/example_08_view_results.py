"""Example 8: Run EDA and open the interactive viewer."""

from pathlib import Path
from dataprism import DataPrism, DataLoader

DATA_DIR = Path(__file__).parent

prism = DataPrism(
    max_categories=20,
    calculate_stability=True,
    cohort_column="split",
    baseline_cohort="train",
    comparison_cohort="test",
    time_based_stability=True,
    time_column="onboarding_date",
    time_window_strategy="monthly",
    baseline_period="first",
    comparison_periods="all",
    min_samples_per_period=100,
)

df = DataLoader.load_csv(DATA_DIR / "credit_risk_dataset.csv")
schema = DataLoader.load_schema(DATA_DIR / "credit_risk_schema.json")

prism.analyze(
    data=df,
    schema=schema,
    target_variable="loan_status",
    output_path=DATA_DIR / "tmp/eda_results.json",
)

prism.view()
