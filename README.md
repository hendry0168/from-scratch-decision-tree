# From Scratch Decision Tree

A learning project demonstrates building a **Decision Tree Classifier from scratch in Python**, training it on HR data from **SQL Server**, and comparing it with a scikit-learn baseline.
---

## 🚀 Overview

This project demonstrates how a decision tree algorithm works by building it **from scratch**, without relying on scikit-learn. It includes:

- Recursive splitting of data based on Gini impurity.
- Prediction by traversing the constructed tree.
- Feature importance analysis (based on information gain).
- Accuracy metrics (precision, recall, confusion matrix).
- Visualizations of decision boundaries.
- Clear, commented code to help you understand each step.

---

**Goal:** Predict whether employees are likely to leave the company based on:

- Gender
- Current Level
- Tenure
- Age
- Education Level

**Technologies:**
- Python (Pandas, Numpy)
- Scikit-learn
- SQL Server (via pyodbc)

---

## ⚠️ Limitations

**This is a learning implementation only**, and **cannot be used in production** without further improvements:

- Only works for **binary classification problems** (target must be 0/1).
- **Does not support categorical variables** (all features must be numeric).
- **No pruning** to prevent overfitting.
- **No cross-validation** or automatic hyperparameter tuning.
- No optimization for large datasets (in-memory only).

---

## 🧠 What I Did

- Implemented a Decision Tree Classifier without any ML libraries.
- Wrote code to connect Python to SQL Server to extract HR data.
- Preprocessed the data (feature engineering, stratified splitting).
- Compared predictions and accuracy between my custom model and scikit-learn.
- Evaluated model performance with confusion matrix and accuracy metrics.

---

## 📊 Results

| Model               | Accuracy |
|---------------------|----------|
| Custom Decision Tree| 52.9%    |
| Scikit-learn        | 53.8%    |

Note: The relatively low accuracy reflects the limited predictive power of the available features. This is common in real-world HR datasets.

---

## 📝 What I Learned

- How to implement tree-based classifiers from first principles.
- How to calculate Gini impurity and information gain.
- How to combine SQL Server data pipelines with Python ML workflows.
- The importance of feature engineering and data quality in predictive modeling.

---
