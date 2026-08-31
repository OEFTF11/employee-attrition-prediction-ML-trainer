import numpy as np

from src.modeling import candidate_models, choose_threshold


def test_all_candidate_models_are_pipelines():
    candidates = candidate_models()

    assert set(candidates) == {"Logistic Regression", "Random Forest", "Gradient Boosting"}
    assert all("preprocessor" in pipeline.named_steps for pipeline, _ in candidates.values())


def test_threshold_is_bounded_and_recall_oriented():
    y_true = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([0.05, 0.2, 0.45, 0.4, 0.8])

    threshold = choose_threshold(y_true, probabilities)

    assert 0.1 <= threshold <= 0.9
    assert threshold <= 0.4
