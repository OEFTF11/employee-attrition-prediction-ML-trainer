"""Train, select, calibrate, evaluate, and persist the attrition model."""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib-cache")
)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from src.config import (
    FIGURES_DIR,
    MODEL_FEATURES,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from src.data import load_dataset, model_frame
from src.modeling import candidate_models, choose_threshold

LOGGER = logging.getLogger(__name__)


def metric_row(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Calculate evaluation metrics at a documented threshold."""
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def subgroup_report(
    raw_test: pd.DataFrame,
    y_test: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """Report performance slices for transparency, not causal fairness claims."""
    audit = raw_test[["Age", "Gender", "MaritalStatus"]].copy()
    audit["AgeGroup"] = pd.cut(
        audit["Age"], bins=[17, 29, 39, 49, 59, 100], labels=["18-29", "30-39", "40-49", "50-59", "60+"]
    )
    audit[TARGET] = y_test
    audit["probability"] = probabilities
    rows: list[dict[str, object]] = []
    for dimension in ["AgeGroup", "Gender", "MaritalStatus"]:
        for group, subset in audit.groupby(dimension, observed=True):
            if len(subset) < 10 or subset[TARGET].nunique() < 2:
                continue
            values = metric_row(subset[TARGET], subset["probability"].to_numpy(), threshold)
            rows.append({"dimension": dimension, "group": str(group), "sample_size": len(subset), **values})
    return pd.DataFrame(rows)


def save_figures(y_test: pd.Series, probabilities: np.ndarray, threshold: float) -> None:
    """Save presentation-ready test-set evaluation plots."""
    predictions = (probabilities >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=["Stay", "Attrition"], cmap="Blues", ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=axes[1])
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_pr_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Execute the complete reproducible training workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    frame = load_dataset()
    X, y = model_frame(frame)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    comparisons: list[dict[str, object]] = []
    searches: dict[str, GridSearchCV] = {}

    for name, (pipeline, grid) in candidate_models().items():
        LOGGER.info("Tuning %s", name)
        search = GridSearchCV(
            pipeline,
            grid,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
            cv=cv,
            n_jobs=-1,
            refit="pr_auc",
        )
        search.fit(X_train, y_train)
        searches[name] = search
        comparisons.append(
            {
                "model": name,
                "cv_pr_auc": search.cv_results_["mean_test_pr_auc"][search.best_index_],
                "cv_roc_auc": search.cv_results_["mean_test_roc_auc"][search.best_index_],
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            }
        )

    comparison = pd.DataFrame(comparisons).sort_values("cv_pr_auc", ascending=False)
    winner_name = str(comparison.iloc[0]["model"])
    best_pipeline = searches[winner_name].best_estimator_
    LOGGER.info("Selected %s by cross-validated PR-AUC", winner_name)

    calibrated_template = CalibratedClassifierCV(clone(best_pipeline), method="sigmoid", cv=3)
    out_of_fold_probabilities = cross_val_predict(
        calibrated_template, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    threshold = choose_threshold(y_train.to_numpy(), out_of_fold_probabilities)
    calibrated_model = clone(calibrated_template).fit(X_train, y_train)
    test_probabilities = calibrated_model.predict_proba(X_test)[:, 1]
    metrics = metric_row(y_test, test_probabilities, threshold)

    importance = permutation_importance(
        calibrated_model, X_test, y_test, scoring="average_precision", n_repeats=20, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance_frame = pd.DataFrame(
        {"feature": MODEL_FEATURES, "importance_mean": importance.importances_mean, "importance_std": importance.importances_std}
    ).sort_values("importance_mean", ascending=False)

    comparison.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    importance_frame.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    subgroup_report(frame.loc[X_test.index], y_test, test_probabilities, threshold).to_csv(
        REPORTS_DIR / "subgroup_metrics.csv", index=False
    )
    save_figures(y_test, test_probabilities, threshold)

    metadata = {
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "selected_model": winner_name,
        "selection_metric": "5-fold cross-validated PR-AUC",
        "decision_threshold": threshold,
        "test_metrics": metrics,
        "features": MODEL_FEATURES,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": RANDOM_STATE,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    joblib.dump({"model": calibrated_model, "threshold": threshold, "metadata": metadata}, MODEL_PATH)

    LOGGER.info("Test metrics: %s", {key: round(value, 3) for key, value in metrics.items()})
    LOGGER.info("Saved model to %s", MODEL_PATH)


if __name__ == "__main__":
    main()
