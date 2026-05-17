# Credit Risk Predictive Modelling — Data Preprocessing & EDA

![Python](https://img.shields.io/badge/Python-3.x-blue)
![sklearn](https://img.shields.io/badge/sklearn-SMOTE%20%7C%20Encoding%20%7C%20Scaling-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Overview
End-to-end data preprocessing and exploratory data analysis (EDA) on two 
large-scale credit risk datasets — **Home Credit Default Risk (HCDR)** and 
**Lending Club Loan Data (LCLD)** — each containing 100,000 rows. The goal 
was to prepare high-quality, model-ready data for predicting borrower 
default risk.

---

## Objectives
- Explore and understand the structure and quality of both datasets
- Handle missing values, duplicates, and class imbalance
- Apply appropriate encoding and normalization techniques
- Engineer new features to improve predictive power
- Determine which dataset is more suitable for downstream modeling

---

## Datasets
| Dataset | Rows | Columns | Target Variable |
|---------|------|---------|----------------|
| Lending Club Loan Data (LCLD) — Accepted | 100,000 | 119 | `loan_status` (7 categories) |
| Lending Club Loan Data (LCLD) — Rejected | 100,000 | 8 | `risk_score` |
| Home Credit Default Risk (HCDR) | 100,000 | 107 | `TARGET` (0/1 binary) |

---

## Methods & Tools
| Step | Technique |
|------|-----------|
| EDA | Distribution analysis, correlation heatmap, boxplot, histogram |
| Missing Value Handling | Mean/mode imputation, column dropping |
| Class Imbalance | SMOTE, Random Oversampling, Undersampling, Class Weighting |
| Categorical Encoding | Binary, Ordinal, One-Hot, Frequency Encoding |
| Normalization | StandardScaler, MinMaxScaler |
| Feature Engineering | INCOME_CREDIT_RATIO, INSTALLMENT_RATIO |
| Environment | Jupyter Notebook / Google Colab |

---

## Key Findings
- **HCDR** is recommended as the superior dataset for predictive modeling:
  - Binary target variable (simpler, cleaner classification task)
  - Richer socioeconomic features (education, housing, occupation type)
  - Smoother preprocessing pipeline with fewer structural issues
- **Class imbalance in HCDR**: 92% non-default (0) vs 8% default (1) — 
  addressed using multiple resampling techniques
- **LCLD** had more preprocessing challenges: unstructured categorical 
  variables, multi-class target, and higher missing value complexity
- New engineered features added meaningful signal:
  - `INCOME_CREDIT_RATIO` — borrower's income relative to loan amount
  - `INSTALLMENT_RATIO` — monthly installment relative to annual income
 
---
 
> Raw datasets are not included due to file size. 
> Download from: [Kaggle — Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) 
> and [Kaggle — Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)


