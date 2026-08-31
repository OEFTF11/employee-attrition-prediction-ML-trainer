"""Dataset loading and validation."""

from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, MODEL_FEATURES, SENSITIVE_AUDIT_COLUMNS, TARGET


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the IBM sample dataset and create a numeric target."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. See the README for setup instructions."
        )
    frame = pd.read_csv(path)
    required = set(MODEL_FEATURES + SENSITIVE_AUDIT_COLUMNS + ["Attrition"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    frame[TARGET] = frame["Attrition"].map({"Yes": 1, "No": 0})
    if frame[TARGET].isna().any():
        raise ValueError("Attrition must contain only 'Yes' and 'No'.")
    return frame


def model_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return complete model inputs and target with aligned indices."""
    clean = frame.dropna(subset=MODEL_FEATURES + [TARGET]).copy()
    return clean[MODEL_FEATURES], clean[TARGET].astype(int)
