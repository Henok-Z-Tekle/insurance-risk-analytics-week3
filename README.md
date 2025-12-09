## 📘 Week 3 — Task 4 – Statistical Modeling & Risk-Based Pricing

## Task 4 – Statistical Modeling & Risk-Based Pricing

Task 4 builds predictive models that sit at the core of a risk-based pricing
framework for ACIS. There are two main modelling objectives:

Claim Severity Model (Regression) – predict TotalClaims for policies with a claim.

Risk / Claim Occurrence Model (Classification) – predict probability of a claim.

Together, these models can support a conceptual formula:

Premium ≈ (Pr(Claim) × Predicted Severity) + Expense Loading + Target Margin

4.1 Data preparation & feature engineering

Key steps implemented in the modeling pipeline (again, filenames may differ slightly):

src/modeling/data_prep.py

Handles missing values (imputation / removal with logs)

Performs type casting and sanity checks

Splits data into:

severity subset: policies with TotalClaims > 0

binary label for claim occurrence: has_claim = (TotalClaims > 0)

src/modeling/features.py

Feature engineering (examples):

loss_ratio = TotalClaims / TotalPremium (capped)

premium_per_month (if policy term is available)

one-hot encoding of Province, ZipCode, VehicleType, etc.

Produces X_train, X_test, y_train, y_test for both tasks.

Train/test split is typically 70/30 with a stratified split for the classification task.

4.2 Models implemented

Severity (Regression)

LinearRegression

RandomForestRegressor

XGBRegressor (gradient boosted trees)

Risk / Claim Occurrence (Classification)

LogisticRegression

RandomForestClassifier

XGBClassifier

The orchestration is handled in:

src/modeling/train_models.py – training & evaluation logic

scripts/run_modeling.py – CLI script to trigger both pipelines and save results.

4.3 Evaluation & comparison

Regression metrics:

RMSE – penalises large claim prediction errors.

MAE – average absolute error.

R² – proportion of variance explained.

Classification metrics:

Accuracy

Precision, Recall, F1-score

ROC-AUC

After running python scripts/run_modeling.py, summary tables are written to:

reports/modeling/
  ├─ regression_summary.csv      # per-model RMSE, MAE, R²
  ├─ classification_summary.csv  # accuracy, precision, recall, F1, ROC-AUC
  ├─ best_models.pkl             # serialized best regression & classifier


Corresponding plots (saved to reports/figures/modeling/) include:

residual vs. fitted plots for regression

feature importance bar charts

ROC curves for the classification models.

4.4 Feature importance & interpretability (SHAP)

To satisfy the interpretability requirement:

src/modeling/interpretability.py

wraps SHAP for tree-based models (Random Forest / XGBoost)

computes:

global feature importance (mean |SHAP value|)

summary plots showing distribution of SHAP values per feature

force plots / waterfall charts for example policies

Outputs are saved as PNGs to:

reports/figures/interpretability/
  ├─ shap_summary_claim_severity.png
  ├─ shap_summary_claim_risk.png
  ├─ shap_example_policy_*.png


These plots highlight the top 5–10 most influential features, e.g.:

Vehicle age or value

Province / ZipCode

Previous claims history

TotalPremium bands

4.5 How to run Task 4 end-to-end

From the project root:

# 1. Ensure cleaned data exists
python -m src.data.preprocess

# 2. Train and evaluate models
python scripts/run_modeling.py


This will:

Load data/processed/insurance_clean.csv

Perform feature engineering and train/test split

Train all regression and classification models

Select best models based on RMSE (regression) and F1-score (classification)

Generate evaluation tables and plots

Run SHAP on the best tree-based models and save interpretability visuals.

4.6 Using the results for business decisions

Some practical ways ACIS can use these outputs:

Risk segmentation – classify policies into low/medium/high risk segments based on predicted claim probability and severity.

Pricing levers – identify features (e.g., vehicle value, zip code, cover type) that most strongly drive claim cost and adjust premiums or underwriting rules.

Marketing strategy – target acquisition campaigns at profiles with high predicted premium and relatively low predicted claims (high margin segment).

Portfolio monitoring – re-train models regularly with updated data, track changes in feature importance, and feed this into pricing committee discussions.

Tip: If you update filenames or add notebooks, just adjust the paths in these sections.
The key is that Task 3 covers hypothesis tests & risk drivers, and Task 4 covers
predictive models, evaluation, and interpretability in a way that matches the rubric.


You can now paste this directly into your `README.md` and tweak any file paths if they differ slightly from your current structure.
::contentReference[oaicite:0]{index=0}
