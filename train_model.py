# train_model.py
#
# Trains a logistic regression model to predict voluntary employee attrition
# using the IBM HR Analytics Employee Attrition & Performance dataset.
#
# Input data file (already in /data): WA_Fn-UseC_-HR-Employee-Attrition.csv
# Output model: models/employee_attrition_model.joblib

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# 1. Load the dataset
# -----------------------------

DATA_PATH = os.path.join("data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = pd.read_csv(DATA_PATH)

# Target: Attrition (Yes = 1, No = 0)
df["Attrition_Flag"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Features we will use in BOTH the model and the Streamlit app
feature_cols = [
    "Age",
    "MonthlyIncome",
    "DistanceFromHome",
    "YearsAtCompany",
    "BusinessTravel",
    "Department",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "JobSatisfaction",
]

X = df[feature_cols].copy()
y = df["Attrition_Flag"].copy()

# Drop any rows with missing values in selected columns (there usually aren’t any)
X = X.dropna()
y = y.loc[X.index]

# -----------------------------
# 2. Build preprocessing + model pipeline
# -----------------------------

numeric_features = [
    "Age",
    "MonthlyIncome",
    "DistanceFromHome",
    "YearsAtCompany",
    "JobSatisfaction",
]

categorical_features = [
    "BusinessTravel",
    "Department",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

log_reg = LogisticRegression(max_iter=1000, class_weight="balanced")  # handle class imbalance

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", log_reg),
    ]
)

# -----------------------------
# 3. Train / test split and training
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

# -----------------------------
# 4. Evaluation
# -----------------------------

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)

print("=== Attrition Model Evaluation ===")
print(f"Accuracy: {acc:.3f}")
print("\nConfusion matrix:")
print(cm)
print("\nClassification report:")
print(report)

# -----------------------------
# 5. Save the trained pipeline
# -----------------------------

os.makedirs("models", exist_ok=True)
MODEL_PATH = os.path.join("models", "employee_attrition_model.joblib")
joblib.dump(model, MODEL_PATH)

print(f"\nSaved trained model pipeline to: {MODEL_PATH}")
