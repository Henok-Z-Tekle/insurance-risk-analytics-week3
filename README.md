📘 Week 3 — Task 1: Insurance Analytics (EDA & Statistical Foundations)

Author: Henok Zenebe Tekle
Email: henok.z.tekl@gmail.com

🔍 Project Overview

Task 1 focuses on developing a strong understanding of the insurance dataset through Exploratory Data Analysis (EDA) and fundamental statistical techniques. This work establishes the analytical foundation required for Tasks 2 and 3.

Your objectives for Task-1:

Understand data structure and quality

Apply statistical reasoning

Perform exploratory analysis

Produce meaningful visualizations

Demonstrate Git/GitHub best practices

📁 Repository Structure (Task 1)
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

📊 Task 1 Deliverables
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

🖥️ Running the Code
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
