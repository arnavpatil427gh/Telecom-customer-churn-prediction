# Telecom Customer Churn Prediction

An end-to-end Machine Learning pipeline utilizing Python, Scikit-Learn, and Imbalanced-Learn to analyze subscriber behavior and predict customer churn in a highly imbalanced dataset representing over 243,000 customers.

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset & Features](#-dataset-overview)
3. [Key Project Phases](#%EF%B8%8F-key-project-phases)
4. [Model Performance Summary](#-model-performance-summary)
5. [Prerequisites & Installation](#-prerequisites--installation)
6. [How to Run](#-how-to-run)
7. [Core Findings](#-core-findings)

---

## 🔍 Project Overview
Customer retention is critical to the profitability of telecom enterprises. This project aims to identify high-risk subscribers prone to churning (`churn = 1`). 

Due to a highly imbalanced target distribution (~80% Non-Churn vs. ~20% Churn), several balancing techniques and model architectures were implemented and compared, including **Logistic Regression**, **Decision Trees**, **Random Forest Classifiers**, and **SMOTE (Synthetic Minority Over-sampling Technique)**.

## 📊 Dataset Overview
The dataset (`telecom_churn.csv`) contains **243,553 rows** and **14 columns** before feature engineering:

* **Demographics**: `gender`, `age`, `num_dependents`, `state`, `city`, `pincode`
* **Account Metadata**: `customer_id`, `telecom_partner` (`Reliance Jio`, `Airtel`, `Vodafone`, `BSNL`), `date_of_registration`
* **Financials**: `estimated_salary`
* **Usage Activity**: `calls_made`, `sms_sent`, `data_used`
* **Target**: `churn` (0: Kept, 1: Churned)

## 🛠️ Key Project Phases

### 1. Data Preprocessing & Cleaning
* Negative values found in physical activity features (`calls_made`, `sms_sent`, `data_used`) were clipped to a logical floor of 0.
* Date fields were parsed into appropriate datetime objects.
* Missing value verification confirmed zero null values.

### 2. Exploratory Data Analysis (EDA)
* Analyzed distributions of demographics and key telecom partners.
* Generated visual distributions of numerical metrics sliced by the churn target.
* Formulated demographic segmentations using binned age groups and salary bands.

### 3. Feature Engineering
* Extracted customer **tenure** in days from the `date_of_registration` field.
* Performed **One-Hot Encoding** on multi-class categorical features (`telecom_partner`, `gender`, `state`, `city`), expanding the feature dimensions to 43 predictors.

### 4. Machine Learning & Resampling
* Established a stratified train-test split (80/20).
* Scaled numeric features appropriately for model compliance.
* Implemented resampling using **SMOTE** to generate synthetic samples for the minority class, addressing severe class-imbalance biases.

---

## 📈 Model Performance Summary

### Baseline Models (Imbalanced Data)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 79.95% | 0.0000 | 0.00% | 0.00% | 0.4983 | 0.1998 |
| **Decision Tree (Depth=8)** | 43.35% | 0.2003 | 61.03% | 30.17% | 0.5008 | 0.2012 |
| **Random Forest** | 72.19% | 0.2075 | 13.73% | 16.53% | 0.5051 | 0.2041 |

### Models Trained on Resampled Data (with SMOTE)

| Model | Accuracy | Recall | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 67.63% | 21.22% | 21.00% |
| **Decision Tree** | 68.46% | 19.91% | 20.00% |
| **Random Forest** | 74.18% | 9.94% | 13.00% |

*Note: The highly consistent 0.50 ROC-AUC and ~0.20 PR-AUC across various paradigms suggest the features in the synthetic telecom data hold extremely weak linear or non-linear correlations with churn behavior, presenting a highly realistic mock feature engineering challenge.*

---

## 📦 Prerequisites & Installation

Make sure you have Python 3.8+ and the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```
2. **Place the dataset:**
   Ensure `telecom_churn.csv` is located in your workspace directory (e.g., `/content/` or the root of the project folder).
3. **Execute the pipeline:**
   Run the Colab notebook cells sequentially or execute the Python script:
   ```bash
   python churn_pipeline.py
   ```

## ⚡ Core Findings
* **Feature Importance**: Behavioral metrics such as `estimated_salary`, `data_used`, and `tenure` emerged as the top structural predictors inside the Random Forest feature importances.
* **SMOTE Impact**: Applying SMOTE helped shift the baseline Logistic Regression model from a trivial "predict majority class only" baseline (Recall of 0.0%) to an active classifier (Recall of 21.22%), though overall feature correlations remained exceptionally flat.

---
