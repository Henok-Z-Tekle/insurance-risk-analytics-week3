## 📘 Week 3 — Task 2 — Data Version Control (DVC)


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
