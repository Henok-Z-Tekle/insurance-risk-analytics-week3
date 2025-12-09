"""Runner script for A/B tests."""
from pathlib import Path
import sys

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ab_tests import load_data, run_all_tests, interpret_results


if __name__ == "__main__":
    data_path = "data/processed/insurance_clean.csv"
    out = "reports/ab_tests"
    df = load_data(data_path)
    results = run_all_tests(df, out)
    interp = interpret_results(results)
    for k, v in interp.items():
        print(k, ":", v)
