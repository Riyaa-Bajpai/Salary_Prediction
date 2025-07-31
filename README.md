# Salary_Prediction
# 💼 Salary Prediction using Ensemble Learning

A machine learning project to predict annual salaries of professionals based on career, education, and workplace features. Built during the IBM PBEL Internship 2025 using the Stack Overflow survey dataset.

## 🧠 Overview

This project explores various factors that influence a software developer’s salary. The goal is to build a robust regression model that can accurately predict salary based on education level, years of experience, job role, remote status, and more.

## ✅ Features

- Salary prediction using ensemble learning
- Data cleaning and category reduction
- Feature encoding with label encoders
- Model tuning with RandomizedSearchCV
- Feature importance analysis and visualizations
- Streamlit-compatible outputs for deployment

## 🛠️ Tech Stack

- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Streamlit
- **Modeling:** RandomForestRegressor (tuned), RandomizedSearchCV
- **Serialization:** Pickle

## 📊 Dataset

- 📌 **Source:** Stack Overflow Developer Survey (via Kaggle)
- 🗓 **Years Covered:** 2020–2023
- 📈 **Records:** 10,000+ salary entries
- 💡 **Target Variable:** `ConvertedCompYearly` (renamed to `Salary`)

## 🧹 Preprocessing

- Removed incomplete or irrelevant rows (e.g., non-full-time workers)
- Handled missing values and outliers (salary bounds: \$12,000–\$250,000)
- Categorical encoding using `LabelEncoder` for:
  - Country
  - Education Level
  - Developer Type
  - Organization Size
  - Remote Work Level
- Applied `log1p` transformation on salary for normality
- Saved cleaned data and encoders to `.pkl` for reuse

## 🤖 Modeling

- **Model Used:** Tuned `RandomForestRegressor`
- **Tuning Method:** `RandomizedSearchCV` with 30 iterations
- **Train-Test Split:** 80-20 ratio
- **Evaluation Metrics:**
  - R² Score: `~0.89`
  - RMSE: `~$15,000`
- **Feature Importance:** Visualized and exported

## 📈 Results

| Metric   | Value     |
|----------|-----------|
| R² Score | ~0.89     |
| RMSE     | ~$15,000  |
| Best Features | Country, Dev Type, Org Size, Remote Work |

## 🚀 Usage

### Clone the repository:
```bash
git clone https://github.com/Riyaa-Bajpai/Salary_Prediction.git
cd Salary_Prediction
