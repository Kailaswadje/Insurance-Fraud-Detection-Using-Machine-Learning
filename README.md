# 🕵️ Insurance Fraud Detection Using Machine Learning

A binary classification system that flags potentially fraudulent insurance claims, helping insurers prioritise investigations and reduce payout losses — built on an imbalanced, real-world-style claims dataset.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-Classification-F7931E?logo=scikitlearn&logoColor=white)
![Imbalanced](https://img.shields.io/badge/Challenge-Class%20Imbalance-red)
![Domain](https://img.shields.io/badge/Domain-InsurTech-blueviolet)

---

## 📌 Business Problem

Fraudulent claims cost insurers billions annually, but investigating every claim is impossible. The goal: build a model that **ranks claims by fraud probability**, so investigation teams focus effort where it matters. In this domain, **recall on the fraud class is the metric that counts** — a missed fraud costs far more than a false alarm.

---

## 📊 Dataset

Claim-level records combining:

- **Policy details** — policy tenure, premium, coverage type, deductible
- **Insured profile** — age, occupation, hobbies, relationship
- **Incident details** — type, severity, time, location, witnesses, police report
- **Claim amounts** — injury, property, and vehicle claim components
- **Target** — `fraud_reported` (Y/N)

---

## 🔬 Approach

1. **EDA** — fraud rate across incident types, severities, and claim amounts
2. **Data cleaning** — handling '?' placeholder values, dropping leakage-prone columns
3. **Feature engineering** — encoding categoricals, deriving claim ratios
4. **Class imbalance handling** — evaluated resampling and class-weighting strategies
5. **Model comparison** — Logistic Regression vs tree-based models (Decision Tree / Random Forest)
6. **Evaluation** — confusion matrix, precision, **recall**, F1, ROC-AUC — not just accuracy

---

## 📈 Results

| Model | Accuracy | Recall (Fraud) | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | XX% | XX% | X.XX | X.XX |
| Random Forest | XX% | XX% | X.XX | X.XX |

> On imbalanced data, a model predicting "no fraud" for everything scores high accuracy while catching zero fraud. That's why this project evaluates on **recall and ROC-AUC** instead.

---

## 💡 Key Insights

- Incident severity and claim amount patterns are strong fraud signals
- Claims lacking police reports or witnesses show elevated fraud rates
- Certain hobby/occupation combinations correlate with fraud in the data — a reminder to audit features for fairness before production use

---

## 🛠️ Tech Stack

Python · Pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter Notebook

---

## 🚀 How to Run

```bash
git clone https://github.com/Kailaswadje/Insurance-Fraud-Detection-Using-Machine-Learning.git
cd Insurance-Fraud-Detection-Using-Machine-Learning
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```

---

## 👤 Author

**Kailas Wadje** — MSc Data Science & AI, University of Liverpool
[GitHub](https://github.com/Kailaswadje) · [LinkedIn](https://www.linkedin.com/in/kwadaje/)

⭐ Star the repo if you found it useful!
