# Student Dropout Prediction System

A machine learning project that predicts whether a student is at risk of dropping out, using Logistic Regression. Built as an early-warning system to help educational institutions identify at-risk students and intervene proactively.

## Problem Statement

Educational institutions typically handle student dropout reactively — no intervention happens until a student has already withdrawn. Early warning signs such as declining grades, unpaid tuition, or lack of financial support are rarely tracked systematically.

This project builds a predictive model that flags at-risk students early, so institutions can offer counseling, financial aid, or academic support before it's too late.

## Dataset

- **Source:** [UCI Machine Learning Repository — Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- **Records:** 4,424 students
- **Features:** 36 (academic, demographic, socioeconomic, enrollment-related)
- **Target:** `Target` (Dropout / Enrolled / Graduate) — converted to binary (`Dropout_Binary`: 1 = Dropout, 0 = Not Dropout)
- **License:** CC BY 4.0

## Project Workflow

| Phase | Description |
|-------|-------------|
| 1 | Learned Logistic Regression fundamentals (sigmoid function, binary classification) |
| 2 | Collected and explored the raw dataset |
| 3 | Cleaned data — checked missing values/duplicates, encoded categorical features, scaled numeric features |
| 4 | Exploratory Data Analysis — distributions, boxplots, correlation heatmap |
| 5 | Trained a Logistic Regression model (80/20 train-test split) |
| 6 | Evaluated the model — confusion matrix, precision, recall, F1, ROC-AUC |
| 7 | Built a risk-prediction application for new student inputs |
| 8 | Documentation and deployment (this repo) |

## Data Preprocessing

- Verified no missing values (dataset pre-cleaned by source)
- Removed duplicate records
- One-hot encoded nominal categorical features (Marital status, Application mode, Course, Nationality, Mother's/Father's occupation)
- Standardized numeric features (grades, age, admission grade, economic indicators) using `StandardScaler`
- Converted 3-class target into binary `Dropout_Binary`

## Exploratory Data Analysis — Key Findings

- **Semester academic performance** (1st and 2nd semester grades) is the strongest predictor of dropout
- **Financial factors** — unpaid tuition fees and lack of scholarship support — correlate strongly with higher dropout rates
- **Debtor status** is associated with increased dropout risk
- **Age at enrollment** — older/non-traditional students show somewhat higher dropout risk

## Model

- **Algorithm:** Logistic Regression (`scikit-learn`)
- **Train/Test Split:** 80% / 20%, stratified on target class
- **Feature scaling:** StandardScaler applied to numeric features

## Evaluation Results

| Metric | Score |
|--------|-------|
| Accuracy | *fill in from your Cell 29 output* |
| Precision | *fill in* |
| Recall | *fill in* |
| F1-Score | *fill in* |
| ROC-AUC | *fill in* |

Confusion matrix and ROC curve are included in `/notebooks/`.

**Note on False Negatives:** In this application, false negatives (students predicted "not at risk" who actually drop out) are the most costly error type, since these students receive no intervention. Future iterations should consider lowering the classification threshold to improve recall at the cost of some false positives.

## Risk Prediction Application

A simple interface (built with Streamlit, see `app.py`) takes student information as input and returns:
- Predicted class (Dropout / Not Dropout)
- Dropout probability
- Risk category (Low / Medium / High)

## How to Reproduce

```bash
# Clone the repository
git clone https://github.com/<your-username>/student-dropout-prediction.git
cd student-dropout-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Jupyter notebook for full pipeline (data cleaning -> EDA -> training -> evaluation)
jupyter notebook notebooks/student_dropout_prediction.ipynb

# Run the Streamlit app
streamlit run app.py
```

## Project Structure

```
student-dropout-prediction/
├── README.md
├── requirements.txt
├── app.py
├── notebooks/
│   └── student_dropout_prediction.ipynb
├── data/
│   └── cleaned_student_dropout.csv
└── model/
    └── logistic_regression_model.pkl
```

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Logistic Regression, preprocessing, evaluation metrics)
- matplotlib, seaborn (visualization)
- Streamlit (deployment)

## Real-World Application

This model can be integrated into a university's student management system as an early-warning system: generating a risk score for every student each semester, so academic advisors can prioritize outreach — counseling, tutoring, or financial aid — toward the students most likely to need it.

## Author

*Your Name* — AI/ML Internship Project
