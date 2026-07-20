# Insurance Fraud Detection — Machine Learning Classification

A machine learning pipeline that detects fraudulent vehicle insurance claims for Global Insure, built on **1,000 historical claims across 40 features**. The project combines class-imbalance handling, RFECV feature selection, and two classifiers — Logistic Regression and a tuned Random Forest — to automate early fraud risk scoring.

## Business Objective

Global Insure loses significant money to fraudulent claims that are only caught late, after payouts are made. This project analyses historical claim, policy, and incident data to classify incoming claims as fraudulent or legitimate, so high-risk claims can be fast-tracked to investigators while genuine customers get faster approvals.

## Dataset

| Property | Detail |
|---|---|
| Records | 1,000 claims × 40 columns |
| Target | `fraud_reported` — 24.7% fraudulent (247 Y / 753 N) — imbalanced |
| Missing data | `property_damage` (360 '?'), `police_report_available` (343 '?'), `collision_type` (178 '?'); fully-null `_c39` column dropped |
| Feature types | Policy details, insured demographics, incident attributes, claim amounts |

## Key EDA Findings

- Fraudulent claims average **$60,302** vs **$50,289** for legitimate ones (~20% higher)
- **Major Damage incidents carry a 60.5% fraud rate**, versus just 10.7% for Minor Damage and 6.7% for Trivial Damage — incident severity is the single strongest signal
- Engineered features such as `days_between_policy_and_incident` and age groups exposed suspicious timing patterns

## Methodology

1. **Cleaning** — replaced '?' placeholders, dropped null/low-variance and leakage-prone columns
2. **Train–Validation Split** then **RandomOverSampler** applied to training data to correct the 75/25 class imbalance
3. **Feature Engineering** — dummy encoding, rare-category grouping, scaling, derived time-based features
4. **Feature Selection** — RFECV with 5-fold StratifiedKFold; `incident_severity` dummies ranked top
5. **Model 1: Logistic Regression** — with optimal-cutoff analysis via sensitivity–specificity and precision–recall curves
6. **Model 2: Random Forest** — GridSearchCV over 576 hyperparameter combinations (2,880 fits); best: `n_estimators=100, max_depth=20, class_weight='balanced_subsample'`; 5-fold CV accuracy ≈ 90.6%

## Model Performance

| Metric | Logistic Regression (validation) | Random Forest (test, tuned) |
|---|---|---|
| Accuracy | 81.0% | 79.0% |
| Sensitivity (Recall) | 67.6% | 55.4% |
| Specificity | 85.4% | 86.7% |
| Precision | 60.2% | 57.8% |
| F1 Score | 0.637 | 0.566 |

Hyperparameter tuning lifted Random Forest recall from **0.27 to 0.55** and F1 from **0.30 to 0.56**. Threshold tuning offers further recall gains where the business cost of missed fraud outweighs false alarms.

## Top Predictors of Fraud

`incident_severity` · `total_claim_amount` · `policy_annual_premium` · `days_between_policy_and_incident` · `incident_hour_of_the_day`

## Business Impact

- **Fraud loss reduction** — flag high-value suspicious claims before payout
- **Operational efficiency** — auto-triage claims so investigators focus on high-risk cases
- **Faster approvals** — low-risk claims clear quicker, improving customer satisfaction
- **Regulatory compliance** — interpretable feature importances support audit and due-diligence requirements

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` (LogisticRegression, RandomForestClassifier, RFECV, GridSearchCV, StratifiedKFold) · `imbalanced-learn` (RandomOverSampler) · `seaborn` / `matplotlib`

## Repository Structure

```
├── Fraudulent_Claim_Detection.ipynb   # Full analysis notebook
├── insurance_claims.csv               # Dataset (1,000 claims)
├── Fraudulent_Detection_Report.pdf    # Summary report with findings
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
jupyter notebook Fraudulent_Claim_Detection.ipynb
```

---

*Author: Kailas Wadje — MSc Data Science & AI, University of Liverpool*
