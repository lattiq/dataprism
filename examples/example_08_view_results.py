"""Example 8: Run EDA and open the interactive viewer."""

from pathlib import Path
from dataprism import EDARunner, DataLoader
from dataprism.viewer.server import serve_results

DATA_DIR = Path(__file__).parent

runner = EDARunner(
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

results = runner.run(
    data=df,
    schema=schema,
    target_variable="loan_status",
)

serve_results(results)
