"""Model definitions and decision-threshold selection."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RANDOM_STATE


def preprocessor() -> ColumnTransformer:
    """Build leakage-safe preprocessing shared by all candidates."""
    return ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def candidate_models() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    """Return candidate pipelines and compact tuning grids."""
    estimators = {
        "Logistic Regression": (
            LogisticRegression(
                class_weight="balanced", max_iter=2_000, random_state=RANDOM_STATE
            ),
            {"classifier__C": [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
            ),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 8],
                "classifier__min_samples_leaf": [1, 4],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [1, 2],
            },
        ),
    }
    return {
        name: (
            Pipeline([("preprocessor", preprocessor()), ("classifier", estimator)]),
            grid,
        )
        for name, (estimator, grid) in estimators.items()
    }


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Choose a threshold maximizing F2, emphasizing attrition recall."""
    thresholds = np.linspace(0.1, 0.9, 161)
    scores = [
        fbeta_score(y_true, probabilities >= threshold, beta=2, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])
