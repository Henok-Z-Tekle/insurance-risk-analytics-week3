"""Exploratory data analysis utilities for the insurance dataset.

Produces descriptive statistics, missing-value report, univariate and
bivariate plots, and simple outlier detection. Designed to run as a
script from the project root so it reads data from `data/processed`.
"""
from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_data(path: str) -> pd.DataFrame:
    """Load CSV data into a pandas DataFrame."""
    return pd.read_csv(path)


def summarize_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    return df.describe(include="number").transpose()


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with counts and percentage of missing values."""
    missing = df.isna().sum()
    pct = 100 * missing / len(df)
    return pd.DataFrame({"missing_count": missing, "missing_pct": pct})


def univariate_plots(df: pd.DataFrame, outdir: str) -> None:
    """Generate histograms for numerical and bar plots for categorical cols."""
    os.makedirs(outdir, exist_ok=True)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in num_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_{col}.png"))
        plt.close()

    for col in cat_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(y=col, data=df)
        plt.title(f"Counts of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"bar_{col}.png"))
        plt.close()


def bivariate_analysis(df: pd.DataFrame, outdir: str, sample: int = 1000) -> None:
    """Create a correlation matrix and a sample of scatter plots for numeric cols."""
    os.makedirs(outdir, exist_ok=True)
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag")
        plt.title("Correlation matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "correlation_matrix.png"))
        plt.close()

        # Pair some columns (limit plotting to avoid heavy outputs)
        subset = df[num_cols].sample(n=min(len(df), sample), random_state=0)
        sns.pairplot(subset)
        plt.savefig(os.path.join(outdir, "pairplot_sample.png"))
        plt.close()


def outlier_detection(df: pd.DataFrame, outdir: str) -> None:
    """Generate boxplots for numeric features to visualize outliers."""
    os.makedirs(outdir, exist_ok=True)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    for col in num_cols:
        plt.figure(figsize=(6, 3))
        sns.boxplot(x=df[col].dropna())
        plt.title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"box_{col}.png"))
        plt.close()


def run_eda(data_path: str = "data/processed/insurance_clean.csv", output_dir: str = "reports/figures") -> Tuple[str, str]:
    """Run EDA pipeline and write summary files to disk.

    Returns (summary_path, missing_path)
    """
    df = load_data(data_path)
    os.makedirs(output_dir, exist_ok=True)

    summary = summarize_data(df)
    summary_path = os.path.join(output_dir, "summary_numeric.csv")
    summary.to_csv(summary_path)

    missing = missing_value_report(df)
    missing_path = os.path.join(output_dir, "missing_report.csv")
    missing.to_csv(missing_path)

    univariate_plots(df, output_dir)
    bivariate_analysis(df, output_dir)
    outlier_detection(df, output_dir)

    return summary_path, missing_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EDA for insurance dataset")
    parser.add_argument("--data", default="data/processed/insurance_clean.csv")
    parser.add_argument("--out", default="reports/figures")
    args = parser.parse_args()
    s, m = run_eda(args.data, args.out)
    print(f"Wrote summary to: {s}")
    print(f"Wrote missing report to: {m}")
