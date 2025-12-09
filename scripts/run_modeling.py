"""Runner for modeling pipeline."""
from pathlib import Path
import sys

# Ensure repo root in path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.modeling import run_modeling


if __name__ == "__main__":
    run_modeling(data_path="data/processed/insurance_clean.csv", out_dir="reports/modeling")
