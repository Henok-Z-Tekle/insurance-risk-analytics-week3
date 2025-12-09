# 📘 Week 3 - Task 3 – A/B Hypothesis Testing (Risk Drivers)
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
