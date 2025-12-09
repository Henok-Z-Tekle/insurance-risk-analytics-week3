# src/run_modeling.py

"""
Run Task 4 – Statistical Modeling end-to-end.

Usage (from project root, venv active):

    python -m src.run_modeling

Outputs:
    - data/processed/model_metrics_regression.csv
    - data/processed/model_metrics_classification.csv
    - data/processed/model_shap_importance.csv  (if shap installed)
"""

import pathlib
import sys

import pandas as pd

from .modeling import (
    train_regression_models,
    train_classification_models,
    compute_shap_importance_tree_model,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed" / "insurance_clean.csv"

OUT_REG_METRICS = PROJECT_ROOT / "data" / "processed" / "model_metrics_regression.csv"
OUT_CLS_METRICS = PROJECT_ROOT / "data" / "processed" / "model_metrics_classification.csv"
OUT_SHAP_IMPORTANCE = PROJECT_ROOT / "data" / "processed" / "model_shap_importance.csv"


def main() -> int:
    if not DATA_PROCESSED.exists():
        print(
            f"[ERROR] Processed dataset not found at {DATA_PROCESSED}. "
            "Run Task 1 & 2 preprocessing first.",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(DATA_PROCESSED)

    # -----------------------------
    # 1. Claim severity regression
    # -----------------------------
    reg_metrics, reg_models = train_regression_models(df)
    OUT_REG_METRICS.parent.mkdir(parents=True, exist_ok=True)
    reg_metrics.to_csv(OUT_REG_METRICS, index=False)
    print("\n=== Regression model metrics ===\n")
    print(reg_metrics.to_string(index=False))
    print(f"\n[INFO] Saved regression metrics to {OUT_REG_METRICS}")

    # -----------------------------
    # 2. Claim occurrence classification
    # -----------------------------
    cls_metrics, cls_models = train_classification_models(df)
    cls_metrics.to_csv(OUT_CLS_METRICS, index=False)
    print("\n=== Classification model metrics ===\n")
    print(cls_metrics.to_string(index=False))
    print(f"\n[INFO] Saved classification metrics to {OUT_CLS_METRICS}")

    # -----------------------------
    # 3. SHAP feature importance
    #    (on best tree-based model by R2 for regression)
    # -----------------------------
    try:
        # pick best regression tree model by R2 if available
        tree_candidates = reg_metrics[
            reg_metrics["model"].isin(["random_forest", "xgboost"])
        ]
        if tree_candidates.empty:
            print("\n[WARN] No tree-based regression model found for SHAP; skipping.")
        else:
            best_model_name = tree_candidates.sort_values("R2", ascending=False)["model"].iloc[0]
            best_model = reg_models[best_model_name]

            # use a small sample for SHAP to keep it fast
            df_sample = df.sample(
                n=min(500, len(df)), random_state=42
            ).reset_index(drop=True)
            X_sample = df_sample.drop(columns=["TotalClaims"])

            shap_importance = compute_shap_importance_tree_model(
                fitted_pipeline=best_model,
                X_sample=X_sample,
                top_n=10,
            )
            shap_importance.to_csv(OUT_SHAP_IMPORTANCE, index=False)
            print("\n=== SHAP feature importance (top 10) ===\n")
            print(shap_importance.to_string(index=False))
            print(f"\n[INFO] Saved SHAP importance to {OUT_SHAP_IMPORTANCE}")
    except ImportError as e:
        print(f"\n[WARN] SHAP not available: {e}. Skipping interpretability step.")
    except Exception as e:
        print(f"\n[WARN] Could not compute SHAP importance: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
