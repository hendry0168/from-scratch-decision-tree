# From Scratch Decision Tree

A learning-focused Python implementation of a decision tree classifier for numeric binary classification problems.

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

## ✨ Key Features

✅ Train a decision tree classifier on numeric data with exactly **two target classes (0/1).**  
✅ Visualize how the algorithm splits the feature space.  
✅ Analyze which features contributed most to decisions.  
✅ Evaluate performance with accuracy, precision, recall, and confusion matrix.

---

## ⚠️ Limitations

**This is a learning implementation only**, and **cannot be used in production** without further improvements:

- Only works for **binary classification problems** (target must be 0/1).
- **Does not support categorical variables** (all features must be numeric).
- **No pruning** to prevent overfitting.
- **No cross-validation** or automatic hyperparameter tuning.
- No optimization for large datasets (in-memory only).

---

## 🧩 Visualizations

The project includes examples to:

- Plot **decision boundaries** using Matplotlib.
- Show **feature importance** as bar charts.
- Display **confusion matrix** for model evaluation.

---

## 📝 Usage

1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn
