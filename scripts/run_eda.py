"""Small runner script to execute the EDA module.

This adjusts `sys.path` so the `src` package is importable when the script
is executed directly (e.g. `python scripts\run_eda.py`).
"""
from pathlib import Path
import sys

# Ensure repository root is on sys.path so `from src...` imports work
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eda.eda import run_eda


if __name__ == "__main__":
    # default paths align with repository layout
    run_eda(data_path="data/processed/insurance_clean.csv", output_dir="reports/figures")
