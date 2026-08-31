"""Shared paths, features, and reproducibility settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "employee_attrition_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TARGET = "Attrition_Flag"

NUMERIC_FEATURES = [
    "Age",
    "MonthlyIncome",
    "DistanceFromHome",
    "YearsAtCompany",
    "JobSatisfaction",
]
CATEGORICAL_FEATURES = [
    "BusinessTravel",
    "Department",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SENSITIVE_AUDIT_COLUMNS = ["Age", "Gender", "MaritalStatus"]
