"""Small runner script to execute the EDA module."""
from src.eda.eda import run_eda


if __name__ == "__main__":
    # default paths align with repository layout
    run_eda(data_path="data/processed/insurance_clean.csv", output_dir="reports/figures")
