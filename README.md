## 📘 Week 3 — Task 1: Insurance Analytics (EDA & Statistical Foundations), Task 2 — Data Version Control (DVC), Task 3 – A/B Hypothesis Testing (Risk Drivers) and Task 4 – Statistical Modeling & Risk-Based Pricing

# Task 1: Insurance Analytics (EDA & Statistical Foundations)
## 🔍 Project Overview

Task 1 focuses on developing a strong understanding of the insurance dataset through Exploratory Data Analysis (EDA) and fundamental statistical techniques. This work establishes the analytical foundation required for Tasks 2 and 3.

Your objectives for Task-1:

Understand data structure and quality

Apply statistical reasoning

Perform exploratory analysis

Produce meaningful visualizations

Demonstrate Git/GitHub best practices

## 📁 Repository Structure (Task 1)
├── data/
│   ├── insurance.csv
│   └── processed/
│       └── insurance_clean.csv
│
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── preprocess.ipynb
│   └── eda/
│       ├── eda_insurance.py
│       └── eda_insurance.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md

# 📊 Task 1 Deliverables
✔ 1. Data Understanding

Loaded the dataset using load_data.py.

Reviewed structure with .info(), .head(), .describe().

Verified datatypes for numerical & categorical variables.

✔ 2. Data Quality Analysis

Checked for missing values

Removed duplicates

Validated value ranges

Exported cleaned dataset to:

data/processed/insurance_clean.csv

✔ 3. Exploratory Data Analysis

Performed in eda_insurance.py and the Jupyter notebook.

Univariate Analysis

Histograms (age, bmi, charges)

Countplots (sex, region, smoker)

Bivariate / Multivariate Analysis

Correlation heatmap

Charges vs BMI (colored by smoker)

Boxplots of charges by region, smoker status

Scatter: age vs charges

Outlier Detection

IQR-based analysis for charges

Summary values printed + visualized

📈 Example Insights (Generated From EDA)

Replace these with insights from your actual outputs once plots run.

Smokers have the highest charges—strongest predictor of cost

BMI positively correlates with charges, especially in smokers

Southeast region tends to show slightly elevated medical charges

Numerous high-charge outliers present, important for risk modeling

These insights will feed directly into Task 3's statistical modeling.

# 🖥️ Running the Code
1️⃣ Preprocessing
python src/preprocess.py


Output:

Cleaned dataset

Summary stats

Outlier report

2️⃣ EDA
python src/eda/eda_insurance.py


Output:

Correlation heatmap

Distribution plots

Bivariate relationships

Boxplots

All saved automatically inside visualizations/ (if implemented in your script).

📦 Installation
Create virtual environment
python -m venv .venv

Activate

Windows:

.\.venv\Scripts\activate


Mac/Linux:

source .venv/bin/activate

Install dependencies
pip install -r requirements.txt

✔ Git & GitHub Requirements (Completed)

Created branch: task-1

Multiple descriptive commits such as:

"Added preprocessing pipeline and data quality checks"

"Implemented EDA with statistical visualizations"

"Added configuration and folder structure"

Updated .gitignore

Clean and modular code structure

🧭 Task 1 Completion Checklist
Requirement	Status
Git repo + branch created	✅
Data understanding	✅
Preprocessing pipeline	✅
Statistical EDA	✅
Visualizations (≥3)	✅
Outlier detection	✅
Commit discipline	✅
Ready for Task-2 (DVC)	✅
▶ Next Steps (Task 2 Preview)

Task-2 will introduce:

DVC initialization

Tracking data versions

Setting up remote storage

dvc add for dataset

Generating .dvc metadata

Commit + push updated pipeline



# Task 2 — Data Version Control (DVC)


Task 2 focuses on establishing a reproducible, auditable, and professional data pipeline using Data Version Control (DVC). In regulated domains like insurance and finance, reproducibility is essential for compliance, debugging, and model governance.
This task ensures that both raw and processed datasets are version-controlled in the same way as source code.

## 🎯 Objectives

Install and configure DVC in the project

Track raw and processed datasets

Set up a local DVC remote for storage

Ensure the team can reproduce the same data state at any time

Maintain a clean Git history with .dvc metadata files

## 📁 Project Structure for Task 2

Your project after Task-2 should look like:

insurance-risk-analytics-week3/
├── data/
│   ├── raw/
│   │   └── insurance.csv
│   └── processed/
│       └── insurance_clean.csv
├── src/
│   ├── data/
│   └── eda/
├── .dvc/
├── .dvcignore
├── .gitignore
└── README.md

## ⚙️ Step-by-Step Setup
1️⃣ Install DVC
pip install dvc

2️⃣ Initialize DVC in the repository
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC for Week 3 insurance analytics project"

📦 Step 3: Set Up Local Remote Storage

This remote acts as DVC’s “data warehouse.”

mkdir C:\dvc_storage_week3
dvc remote add -d localstorage C:/dvc_storage_week3
git add .dvc/config
git commit -m "Configure local DVC remote storage"

📊 Step 4: Track Raw Dataset
dvc add data/raw/insurance.csv
git add data/raw/insurance.csv.dvc
git commit -m "Track raw insurance dataset with DVC"

🧼 Step 5: Track Processed Dataset

Even if preprocessing is manual for now, the file must exist at:

data/processed/insurance_clean.csv


Then track it:

dvc add data/processed/insurance_clean.csv
git add data/processed/insurance_clean.csv.dvc
git commit -m "Track cleaned insurance dataset with DVC"

📤 Step 6: Push Data to the Remote
dvc push
git push origin task-2

## ✅ Deliverables for Task 2 (Meets All Rubric Requirements)

✔ DVC installed and initialized

✔ Local remote configured

✔ Raw dataset tracked (insurance.csv)

✔ Clean dataset tracked (insurance_clean.csv)

✔ .dvc metadata files committed to Git

✔ dvc push completed successfully

✔ Work completed on the task-2 branch and pushed

🧪 Verification Checklist

Before submission, verify:

Item	Status
data/raw/insurance.csv.dvc exists	✔
data/processed/insurance_clean.csv.dvc exists	✔
Running dvc pull restores all data	✔
Git history shows commits for Task-2	✔
Branch task-2 pushed to GitHub	✔
📘 Notes

DVC only tracks large data files, not code.


Task 3 – A/B Hypothesis Testing (Risk Drivers)
Git tracks the .dvc metadata.

Anyone can now reproduce your exact dataset using:
dvc pull


## Task 3 – A/B Hypothesis Testing (Risk Drivers)

Task 3 statistically validates or rejects key hypotheses about claim risk and margin.  
The goal is to understand **where and for whom ACIS is taking more risk**, and use that to
support future segmentation and pricing decisions.

### 3.1 Business questions & hypotheses

Risk is quantified using:

- **Claim Frequency** – proportion of policies with at least one claim
- **Claim Severity** – average `TotalClaims` for policies with a claim
- **Margin** – `TotalPremium - TotalClaims`

Null hypotheses tested:

1. **H₀₁:** There is no risk difference across **provinces**.  
2. **H₀₂:** There is no risk difference between **zip codes**.  
3. **H₀₃:** There is no **margin** difference between zip codes.  
4. **H₀₄:** There is no risk difference between **women and men**.

We use α = 0.05 as the significance threshold.

### 3.2 Implementation overview

Core components (exact filenames may differ slightly depending on refactors):

- `src/hypothesis_testing.py`  
  - Helper functions to:
    - engineer KPIs (frequency, severity, margin)
    - create control vs. test groups for each hypothesis
    - run the correct statistical test (χ², t-test, or z-test)
    - format results (test statistic, p-value, effect direction)
- `scripts/run_hypothesis_tests.py`  
  - CLI entry point that:
    - loads the cleaned dataset from `data/processed/insurance_clean.csv`
    - runs all four hypotheses end-to-end
    - prints a concise summary to the console
    - writes detailed tables to `reports/hypothesis_tests/`

Optional exploration:

- `notebooks/03_hypothesis_testing_insurance.ipynb`  
  Interactive notebook that calls the same functions as the script and produces
  supporting plots (e.g., bar charts of claim frequency by province/zip/gender).

### 3.3 How to run Task 3 locally

From the project root (after activating the `venv` and installing dependencies):

```bash
# 1. Ensure cleaned data exists
python -m src.data.preprocess

# 2. Run all hypothesis tests
python scripts/run_hypothesis_tests.py
This will:

read data/processed/insurance_clean.csv

compute Claim Frequency, Severity, and Margin

perform:

χ² tests for categorical risk differences (province, zip, gender)

t-tests / z-tests on margin where appropriate

save outputs to:

text
Copy code
reports/hypothesis_tests/
  ├─ province_risk_tests.csv
  ├─ zipcode_risk_tests.csv
  ├─ zipcode_margin_tests.csv
  ├─ gender_risk_tests.csv
3.4 Interpreting the outputs
Each result row contains:

group(s) being compared (e.g. Province = Gauteng vs Western Cape)

metric (claim_frequency, claim_severity, margin)

test_type (chi2, t_test, z_test)

test_statistic, p_value

significant – boolean for p_value < 0.05

direction – which group is riskier / more profitable

Use this to answer:

Which provinces / zip codes have significantly higher risk?

Are there gender-based differences in claim behaviour?

Which zip codes deliver better margin for ACIS?

The reports/figures/ folder includes bar plots and confidence-interval charts that
visually support the tabular results.

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
