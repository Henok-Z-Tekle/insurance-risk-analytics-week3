# src/hypothesis_testing.py

"""
Task 3 – A/B Hypothesis Testing for Insurance Risk Analytics

Implements:
- KPI definition:
    * Claim Frequency (proportion with at least one claim)
    * Claim Severity (average size of positive claims)
    * Margin (TotalPremium - TotalClaims)
- Data segmentation for A/B tests
- Statistical tests:
    * Chi-square tests for categorical vs binary outcome
    * Two-sample t-tests for numerical metrics
- Four required hypotheses:
    1. Risk differences across provinces
    2. Risk differences between zip codes
    3. Margin differences between zip-code groups
    4. Risk differences between genders

All functions are pure and testable, with simple error handling.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

@dataclass
class ColumnConfig:
    province: str = "Province"
    zipcode: str = "ZipCode"
    gender: str = "Gender"
    total_premium: str = "TotalPremium"
    total_claims: str = "TotalClaims"
    policy_id: str = "PolicyID"  # optional


CFG = ColumnConfig()


# -------------------------------------------------------------------
# Metric engineering – KPIs
# -------------------------------------------------------------------

def add_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived KPI columns:

    - claim_occurred: 1 if TotalClaims > 0, else 0
    - claim_severity: TotalClaims when > 0, NaN otherwise
    - margin: TotalPremium - TotalClaims

    Returns a copy of the dataframe with new columns.
    """
    missing_cols = {CFG.total_premium, CFG.total_claims} - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns for metrics: {missing_cols}")

    df = df.copy()
    df["claim_occurred"] = (df[CFG.total_claims] > 0).astype(int)
    df["claim_severity"] = np.where(
        df[CFG.total_claims] > 0, df[CFG.total_claims], np.nan
    )
    df["margin"] = df[CFG.total_premium] - df[CFG.total_claims]
    return df


def summarize_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Summarize KPIs by a categorical group (Province, ZipCode, Gender).

    Returns columns:
        group_col, policy_count, claim_frequency,
        avg_severity, avg_margin
    """
    required = {"claim_occurred", "claim_severity", "margin"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Missing metric columns {missing}. "
            "Call add_risk_metrics() before summarising."
        )

    grouped = df.groupby(group_col)
    summary = pd.DataFrame({
        group_col: grouped.size().index,
        "policy_count": grouped.size().values,
        "claim_frequency": grouped["claim_occurred"].mean().values,
        "avg_severity": grouped["claim_severity"].mean().values,
        "avg_margin": grouped["margin"].mean().values,
    })

    return summary


# -------------------------------------------------------------------
# Statistical helper functions
# -------------------------------------------------------------------

def chi_square_from_crosstab(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str = "claim_occurred",
) -> Tuple[float, float, pd.DataFrame]:
    """
    Run chi-square test of independence between:

        - group_col: categorical (Province, ZipCode, Gender)
        - outcome_col: binary (usually claim_occurred)

    Returns:
        chi2_stat, p_value, contingency_table
    """
    if group_col not in df.columns or outcome_col not in df.columns:
        raise KeyError(f"Columns '{group_col}' or '{outcome_col}' not found.")

    contingency = pd.crosstab(df[group_col], df[outcome_col])
    chi2, p, _, _ = stats.chi2_contingency(contingency)

    return chi2, p, contingency


def ttest_between_groups(
    df: pd.DataFrame,
    metric_col: str,
    group_col: str,
    group_a_vals: List,
    group_b_vals: List,
) -> Tuple[float, float, int, int]:
    """
    Independent two-sample t-test for metric_col between
    group A and group B (based on values in group_col).

    Returns:
        t_stat, p_value, n_a, n_b
    """
    if metric_col not in df.columns or group_col not in df.columns:
        raise KeyError(f"Columns '{metric_col}' or '{group_col}' not found.")

    df_valid = df[[metric_col, group_col]].dropna()

    sample_a = df_valid[df_valid[group_col].isin(group_a_vals)][metric_col]
    sample_b = df_valid[df_valid[group_col].isin(group_b_vals)][metric_col]

    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 5 or n_b < 5:
        raise ValueError(
            f"Insufficient data for t-test: "
            f"{n_a} obs in group A, {n_b} in group B."
        )

    t_stat, p = stats.ttest_ind(sample_a, sample_b, equal_var=False)
    return t_stat, p, n_a, n_b


# -------------------------------------------------------------------
# Task-3 Hypotheses
# -------------------------------------------------------------------

def test_risk_across_provinces(
    df: pd.DataFrame, alpha: float = 0.05
) -> Dict:
    """
    H0: There are no risk differences across provinces.
    Risk proxy: claim frequency (claim_occurred).

    Test: chi-square of independence.
    """
    df_metrics = add_risk_metrics(df)
    chi2, p, table = chi_square_from_crosstab(
        df_metrics, group_col=CFG.province, outcome_col="claim_occurred"
    )

    reject = p < alpha
    interpretation = (
        "Reject H0: Claim frequency differs across provinces. "
        "ACIS should consider province-level risk segmentation and "
        "potential regional premium adjustments."
        if reject
        else "Fail to reject H0: No strong evidence of different risk levels "
             "between provinces based on claim frequency."
    )

    return {
        "id": "H1",
        "hypothesis": "No risk differences across provinces",
        "kpi": "Claim frequency",
        "test_type": "Chi-square",
        "alpha": alpha,
        "p_value": p,
        "reject_null": reject,
        "statistic": chi2,
        "business_interpretation": interpretation,
        "details_table": table,
    }


def test_risk_between_zipcodes(
    df: pd.DataFrame, alpha: float = 0.05
) -> Dict:
    """
    H0: There is no risk difference between zip codes.
    Risk proxy: claim frequency.

    Test: chi-square over ZipCode vs claim_occurred.
    """
    df_metrics = add_risk_metrics(df)
    chi2, p, table = chi_square_from_crosstab(
        df_metrics, group_col=CFG.zipcode, outcome_col="claim_occurred"
    )

    reject = p < alpha
    interpretation = (
        "Reject H0: Claim frequency varies across zip codes. "
        "ACIS can use zip-code level risk bands for more accurate pricing."
        if reject
        else "Fail to reject H0: Zip-code differences do not show a significant "
             "impact on claim frequency at α = 0.05."
    )

    return {
        "id": "H2",
        "hypothesis": "No risk differences between zip codes",
        "kpi": "Claim frequency",
        "test_type": "Chi-square",
        "alpha": alpha,
        "p_value": p,
        "reject_null": reject,
        "statistic": chi2,
        "business_interpretation": interpretation,
        "details_table": table,
    }


def test_margin_between_zip_groups(
    df: pd.DataFrame,
    group_a_zips: List,
    group_b_zips: List,
    alpha: float = 0.05,
) -> Dict:
    """
    H0: No difference in margin (TotalPremium - TotalClaims)
        between zip-code Group A and Group B.

    This is the most "A/B" style test: we explicitly form control and test
    groups based on zip-code sets chosen from EDA (e.g., low-risk vs high-risk).

    Test: two-sample t-test on margin.
    """
    df_metrics = add_risk_metrics(df)

    t_stat, p, n_a, n_b = ttest_between_groups(
        df_metrics,
        metric_col="margin",
        group_col=CFG.zipcode,
        group_a_vals=group_a_zips,
        group_b_vals=group_b_zips,
    )

    reject = p < alpha
    interpretation = (
        "Reject H0: Margin differs significantly between the selected "
        "zip-code groups. ACIS can target the higher-margin group for "
        "marketing campaigns or consider premium adjustments for the "
        "lower-margin group."
        if reject
        else "Fail to reject H0: The selected zip-code groups do not show a "
             "statistically significant difference in margin."
    )

    return {
        "id": "H3",
        "hypothesis": "No margin difference between zip-code groups",
        "kpi": "Margin (TotalPremium - TotalClaims)",
        "test_type": "Two-sample t-test",
        "alpha": alpha,
        "p_value": p,
        "reject_null": reject,
        "statistic": t_stat,
        "group_a_size": n_a,
        "group_b_size": n_b,
        "group_a_zips": group_a_zips,
        "group_b_zips": group_b_zips,
        "business_interpretation": interpretation,
    }


def test_risk_between_genders(
    df: pd.DataFrame, alpha: float = 0.05
) -> Dict:
    """
    H0: No significant risk difference between women and men.
    Risk proxy: claim frequency.

    Test: chi-square over Gender vs claim_occurred.
    """
    df_metrics = add_risk_metrics(df)
    chi2, p, table = chi_square_from_crosstab(
        df_metrics, group_col=CFG.gender, outcome_col="claim_occurred"
    )

    reject = p < alpha
    interpretation = (
        "Reject H0: Claim frequency differs between genders. "
        "Gender appears to be associated with risk; ACIS should consider "
        "how this interacts with fairness and regulatory constraints."
        if reject
        else "Fail to reject H0: No strong evidence that risk differs between "
             "genders based on claim frequency."
    )

    return {
        "id": "H4",
        "hypothesis": "No risk differences between genders",
        "kpi": "Claim frequency",
        "test_type": "Chi-square",
        "alpha": alpha,
        "p_value": p,
        "reject_null": reject,
        "statistic": chi2,
        "business_interpretation": interpretation,
        "details_table": table,
    }


def run_all_hypothesis_tests(
    df: pd.DataFrame,
    margin_group_a_zips: List,
    margin_group_b_zips: List,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Convenience function: run all four hypotheses and return a tidy summary
    DataFrame suitable for saving to CSV and referencing in the report.

    Contingency tables are omitted from the returned table to keep it compact.
    """
    results = [
        test_risk_across_provinces(df, alpha),
        test_risk_between_zipcodes(df, alpha),
        test_margin_between_zip_groups(df, margin_group_a_zips, margin_group_b_zips, alpha),
        test_risk_between_genders(df, alpha),
    ]

    summary_rows = []
    for r in results:
        summary_rows.append({
            "id": r["id"],
            "hypothesis": r["hypothesis"],
            "kpi": r["kpi"],
            "test_type": r["test_type"],
            "alpha": r["alpha"],
            "p_value": r["p_value"],
            "reject_null": r["reject_null"],
            "business_interpretation": r["business_interpretation"],
        })

    return pd.DataFrame(summary_rows)
