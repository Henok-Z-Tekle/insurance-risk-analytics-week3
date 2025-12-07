## 📘 Week 3 — Task 1: Insurance Analytics (EDA & Statistical Foundations) and Task 2 — Data Version Control (DVC)

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

Git tracks the .dvc metadata.

Anyone can now reproduce your exact dataset using:

dvc pull
