# Heart Disease Prediction 🫀

Machine learning project that predicts the presence of heart disease in patients based on clinical and demographic data. Built as part of the **CodeAlpha Machine Learning Internship** — Task 4: Disease Prediction from Medical Data.

## Overview

This project applies classification algorithms to the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction), comparing three models — Logistic Regression, Random Forest, and Support Vector Machine — and evaluating them with Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

**Best model:** SVM, achieving **0.939 ROC-AUC** on the held-out test set.

## Dataset

- 918 patient records, 11 clinical features
- Combines 5 well-known heart disease datasets (Cleveland, Hungarian, Switzerland, Long Beach VA, Statlog)
- Target: `HeartDisease` (1 = disease, 0 = normal)

| Feature | Description |
|---|---|
| Age | Age of the patient (years) |
| Sex | M = Male, F = Female |
| ChestPainType | TA, ATA, NAP, or ASY |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mm/dl) |
| FastingBS | 1 if fasting blood sugar > 120 mg/dl, else 0 |
| RestingECG | Normal, ST, or LVH |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Y/N — exercise-induced angina |
| Oldpeak | ST depression induced by exercise |
| ST_Slope | Up, Flat, or Down |

## Approach

1. **Data quality check** — detected disguised missing values (Cholesterol/RestingBP recorded as 0, a biological impossibility) and imputed them using the per-class median instead of dropping ~19% of the data.
2. **Preprocessing** — one-hot encoded categorical features; split into train/test (80/20, stratified) **before** scaling to avoid data leakage.
3. **Modeling** — trained Logistic Regression, Random Forest, and SVM on the same data.
4. **Evaluation** — compared all models on Accuracy, Precision, Recall, F1, ROC-AUC, plus confusion matrices, ROC curves, and 5-fold cross-validation to confirm stability.
5. **Feature importance** — identified ST_Slope, MaxHR, and Oldpeak as the strongest predictors, consistent with established cardiology indicators.

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8913 | 0.8942 | 0.9118 | 0.9029 | 0.9350 |
| Random Forest | 0.8913 | 0.8942 | 0.9118 | 0.9029 | 0.9364 |
| **SVM (best)** | 0.8641 | 0.8738 | 0.8824 | 0.8780 | **0.9388** |

5-fold cross-validation accuracy: Logistic Regression 84.2%, Random Forest 84.3%, SVM 83.7% — confirming results generalize well beyond a single train/test split.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (Logistic Regression, Random Forest, SVM, preprocessing, metrics)
- matplotlib, seaborn (visualization)

## Files

- `Heart_Disease_Prediction.ipynb` — full notebook (EDA, preprocessing, modeling, evaluation)
- `heart.csv` — dataset used


