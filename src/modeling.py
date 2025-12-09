# src/modeling.py

"""
Task 4 – Statistical Modeling for Insurance Risk Analytics

This module builds and evaluates predictive models for:

1. Claim Severity (regression)
   - Target: TotalClaims on policies with a claim > 0
   - Models: Linear Regression, Random Forest, XGBoost (if installed)
   - Metrics: RMSE, R-squared

2. Claim Occurrence (classification, advanced but useful)
   - Target: claim_occurred (binary)
   - Models: Logistic Regression, Random Forest, XGBoost (if installed)
   - Metrics: Accuracy, Precision, Recall, F1-score

It also:
- Handles basic missing values
- Encodes categorical variables
- Performs train-test split
- Computes SHAP feature importance for the best tree-based model

The functions are modular so they can be reused from notebooks or scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# XGBoost is optional – handle gracefully if not installed
try:
    from xgboost import XGBRegressor, XGBClassifier  # type: ignore
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# SHAP for model interpretability (tree/linear models)
try:
    import shap  # type: ignore
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# --------------------------------------------------------------------
# Basic config
# --------------------------------------------------------------------

@dataclass
class ModelingConfig:
    target_claims: str = "TotalClaims"
    target_premium: str = "TotalPremium"
    # Column engineered in Task 3 – if not present, we will create it here
    target_claim_occurred: str = "claim_occurred"
    test_size: float = 0.2
    random_state: int = 42


CFG = ModelingConfig()


# --------------------------------------------------------------------
# Utility: Ensure risk metrics/targets exist
# --------------------------------------------------------------------

def add_claim_occurred_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary claim_occurred column if missing."""
    if CFG.target_claim_occurred not in df.columns:
        if CFG.target_claims not in df.columns:
            raise KeyError(
                f"Column '{CFG.target_claims}' not found; required to derive "
                f"'{CFG.target_claim_occurred}'."
            )
        df = df.copy()
        df[CFG.target_claim_occurred] = (df[CFG.target_claims] > 0).astype(int)
    return df


# --------------------------------------------------------------------
# Utility: feature / target split + preprocessing
# --------------------------------------------------------------------

def split_features_target(
    df: pd.DataFrame, target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer with:
    - Numeric: median imputation + scaling
    - Categorical: most-frequent imputation + one-hot encoding
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# --------------------------------------------------------------------
# Regression: Claim Severity (TotalClaims on claims > 0)
# --------------------------------------------------------------------

def prepare_severity_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to policies where TotalClaims > 0 and drop extreme missing data if any.
    """
    if CFG.target_claims not in df.columns:
        raise KeyError(f"Column '{CFG.target_claims}' not found in dataframe.")

    df_sev = df[df[CFG.target_claims] > 0].copy()
    # Drop rows with all-NaN aside from the target
    df_sev = df_sev.dropna(axis=0, how="all", subset=df_sev.columns.difference([CFG.target_claims]))
    return df_sev


def train_regression_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Train severity models:
        - Linear Regression
        - Random Forest Regressor
        - XGBRegressor (if available)

    Returns:
        metrics_df: DataFrame with RMSE and R2 for each model
        models: dict name -> fitted Pipeline
    """
    df_sev = prepare_severity_dataset(df)
    X, y = split_features_target(df_sev, CFG.target_claims)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG.test_size, random_state=CFG.random_state
    )

    preprocessor = build_preprocessor(X_train)

    models: Dict[str, Pipeline] = {}

    # Linear Regression
    linreg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    linreg.fit(X_train, y_train)
    models["linear_regression"] = linreg

    # Random Forest Regressor
    rf_reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200,
                random_state=CFG.random_state,
                n_jobs=-1,
            )),
        ]
    )
    rf_reg.fit(X_train, y_train)
    models["random_forest"] = rf_reg

    # XGBoost Regressor (optional)
    if HAS_XGBOOST:
        xgb_reg = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", XGBRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=CFG.random_state,
                    objective="reg:squarederror",
                )),
            ]
        )
        xgb_reg.fit(X_train, y_train)
        models["xgboost"] = xgb_reg

    # Evaluate
    rows = []
    for name, pipe in models.items():
        y_pred = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        rows.append(
            {"model": name, "task": "severity_regression", "RMSE": rmse, "R2": r2}
        )

    metrics_df = pd.DataFrame(rows)
    return metrics_df, models


# --------------------------------------------------------------------
# Classification: Claim Occurrence (advanced but aligns with spec)
# --------------------------------------------------------------------

def prepare_occurrence_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure claim_occurred exists and drop rows with all-NaN features.
    """
    df_occ = add_claim_occurred_column(df)
    df_occ = df_occ.dropna(
        axis=0, how="all",
        subset=df_occ.columns.difference([CFG.target_claim_occurred])
    )
    return df_occ


def train_classification_models(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Pipeline]]:
    """
    Train occurrence models:
        - Logistic Regression
        - Random Forest Classifier
        - XGBClassifier (if available)

    Returns:
        metrics_df: DataFrame with accuracy, precision, recall, F1
        models: dict name -> fitted Pipeline
    """
    df_occ = prepare_occurrence_dataset(df)
    X, y = split_features_target(df_occ, CFG.target_claim_occurred)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CFG.test_size, random_state=CFG.random_state, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    models: Dict[str, Pipeline] = {}

    # Logistic Regression
    log_reg = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    log_reg.fit(X_train, y_train)
    models["logistic_regression"] = log_reg

    # Random Forest Classifier
    rf_clf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=CFG.random_state,
                n_jobs=-1,
            )),
        ]
    )
    rf_clf.fit(X_train, y_train)
    models["random_forest_classifier"] = rf_clf

    # XGBoost Classifier (optional)
    if HAS_XGBOOST:
        xgb_clf = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=CFG.random_state,
                    objective="binary:logistic",
                    eval_metric="logloss",
                )),
            ]
        )
        xgb_clf.fit(X_train, y_train)
        models["xgboost_classifier"] = xgb_clf

    # Evaluate
    rows = []
    for name, pipe in models.items():
        y_pred = pipe.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))

        rows.append(
            {
                "model": name,
                "task": "occurrence_classification",
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
            }
        )

    metrics_df = pd.DataFrame(rows)
    return metrics_df, models


# --------------------------------------------------------------------
# SHAP feature importance
# --------------------------------------------------------------------

def compute_shap_importance_tree_model(
    fitted_pipeline: Pipeline,
    X_sample: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Compute SHAP feature importance for a tree-based model
    inside a sklearn Pipeline (preprocess + model).

    Returns a DataFrame with:
        feature_name, mean_abs_shap
    """
    if not HAS_SHAP:
        raise ImportError(
            "The 'shap' package is not installed. "
            "Install it with `pip install shap` to use this function."
        )

    # Separate preprocessor and model
    preprocessor: ColumnTransformer = fitted_pipeline.named_steps["preprocess"]
    model = fitted_pipeline.named_steps["model"]

    # Transform features
    X_trans = preprocessor.transform(X_sample)

    # Get feature names after one-hot + scaling
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            encoder: OneHotEncoder = transformer.named_steps["encoder"]
            feature_names.extend(encoder.get_feature_names_out(cols))

    # Build SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_trans)

    # For binary classification, shap_values is list; take positive class
    if isinstance(shap_values, list):
        shap_vals = shap_values[-1]
    else:
        shap_vals = shap_values

    mean_abs = np.mean(np.abs(shap_vals), axis=0)

    importance = (
        pd.DataFrame({"feature_name": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return importance
