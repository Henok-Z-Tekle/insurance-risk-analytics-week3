"""A/B Hypothesis testing utilities for insurance dataset.

Implements KPI calculation, segmentation, statistical tests (chi-square
for categorical, t-test for numeric), p-value analysis and simple business
interpretation scaffolding. Outputs results to `reports/ab_tests`.
"""
from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def load_data(path: str = "data/processed/insurance_clean.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def compute_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """Compute KPIs per grouping: Claim Frequency, Claim Severity, Margin.

    Expects columns: 'TotalPremium', 'TotalClaims', 'ClaimOccurred' (or
    uses TotalClaims>0 to indicate a claim).
    """
    df = df.copy()
    if "ClaimOccurred" not in df.columns:
        df["ClaimOccurred"] = (df.get("TotalClaims", 0) > 0).astype(int)

    df["Margin"] = df.get("TotalPremium", 0) - df.get("TotalClaims", 0)

    return df


def chi2_test_of_independence(df: pd.DataFrame, group_col: str, target_col: str) -> Tuple[float, float, pd.DataFrame]:
    """Run chi-square test between group_col and categorical target_col.

    Returns (chi2_stat, p_value, contingency_table)
    """
    contingency = pd.crosstab(df[group_col], df[target_col])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return float(chi2), float(p), contingency


def ttest_group_means(df: pd.DataFrame, group_col: str, numeric_col: str, group_a, group_b) -> Tuple[float, float]:
    """Two-sample t-test (unequal variance) between two groups for numeric_col."""
    a = df.loc[df[group_col] == group_a, numeric_col].dropna().astype(float)
    b = df.loc[df[group_col] == group_b, numeric_col].dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return float('nan'), float('nan')
    tstat, p = stats.ttest_ind(a, b, equal_var=False)
    return float(tstat), float(p)


def run_all_tests(df: pd.DataFrame, output_dir: str = "reports/ab_tests") -> Dict[str, Dict]:
    """Run the required four hypothesis tests and write results.

    Tests performed:
    1) Risk across provinces (chi-square on ClaimOccurred vs Province)
    2) Risk between zip codes (chi-square using top-2 zip codes by count)
    3) Margin between zip codes (t-test between two top zip codes)
    4) Risk between genders (chi-square on ClaimOccurred vs Gender)
    """
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Dict] = {}

    # Ensure KPIs
    compute_kpis(df)

    # 1) Province risk (assume column 'Province' exists)
    if "Province" in df.columns:
        chi2, p, tab = chi2_test_of_independence(df, "Province", "ClaimOccurred")
        results["province_risk"] = {"chi2": chi2, "p_value": p}
        tab.to_csv(os.path.join(output_dir, "province_claims_contingency.csv"))
    else:
        results["province_risk"] = {"error": "Province column not found"}

    # 2 & 3) Zip code comparisons: pick top 2 frequent zip codes
    if "ZipCode" in df.columns:
        top_zips = df["ZipCode"].value_counts().nlargest(2).index.tolist()
        if len(top_zips) >= 2:
            # chi-square of claim occurrence across zip codes (full table)
            chi2z, pz, tabz = chi2_test_of_independence(df, "ZipCode", "ClaimOccurred")
            results["zipcode_risk"] = {"chi2": chi2z, "p_value": pz}
            tabz.to_csv(os.path.join(output_dir, "zipcode_claims_contingency.csv"))

            # margin t-test between top two zips
            tstat, p_margin = ttest_group_means(df, "ZipCode", "Margin", top_zips[0], top_zips[1])
            results["zipcode_margin"] = {"zip_a": top_zips[0], "zip_b": top_zips[1], "tstat": tstat, "p_value": p_margin}
        else:
            results["zipcode_risk"] = {"error": "Not enough ZipCode categories"}
    else:
        results["zipcode_risk"] = {"error": "ZipCode column not found"}

    # 4) Gender risk
    gender_col = None
    for candidate in ("Gender", "Sex", "gender", "sex"):
        if candidate in df.columns:
            gender_col = candidate
            break
    if gender_col:
        chg, pg, ttab = chi2_test_of_independence(df, gender_col, "ClaimOccurred")
        results["gender_risk"] = {"chi2": chg, "p_value": pg}
        ttab.to_csv(os.path.join(output_dir, "gender_claims_contingency.csv"))
    else:
        results["gender_risk"] = {"error": "Gender column not found"}

    # Save summary
    summary_df = pd.DataFrame({k: pd.Series(v) for k, v in results.items()}).T
    summary_df.to_csv(os.path.join(output_dir, "ab_test_summary.csv"))
    return results


def interpret_results(results: Dict[str, Dict], alpha: float = 0.05) -> Dict[str, str]:
    """Basic business interpretation for p-values.

    Returns a mapping of test -> interpretation text.
    """
    interp: Dict[str, str] = {}
    for test, res in results.items():
        if "p_value" in res and not np.isnan(res["p_value"]):
            if res["p_value"] < alpha:
                interp[test] = f"Reject null (p={res['p_value']:.4f}). Evidence of difference; consider segmentation or regional premium adjustment."
            else:
                interp[test] = f"Fail to reject null (p={res['p_value']:.4f}). No statistical evidence of difference at alpha={alpha}."
        else:
            interp[test] = "Test not performed or insufficient data"
    return interp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run A/B hypothesis tests on insurance dataset")
    parser.add_argument("--data", default="data/processed/insurance_clean.csv")
    parser.add_argument("--out", default="reports/ab_tests")
    args = parser.parse_args()

    df = load_data(args.data)
    res = run_all_tests(df, args.out)
    interp = interpret_results(res)
    for k, v in interp.items():
        print(k, ":", v)
