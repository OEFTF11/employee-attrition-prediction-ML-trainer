import pandas as pd
import pytest

from src.config import MODEL_FEATURES
from src.data import load_dataset, model_frame


def test_project_dataset_has_expected_model_columns():
    frame = load_dataset()
    X, y = model_frame(frame)

    assert list(X.columns) == MODEL_FEATURES
    assert len(X) == len(y) == 1_470
    assert set(y.unique()) == {0, 1}


def test_loader_rejects_invalid_schema(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"Attrition": ["Yes"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_loader_rejects_unknown_target(tmp_path):
    frame = load_dataset().head(2).copy()
    frame.loc[frame.index[0], "Attrition"] = "Maybe"
    path = tmp_path / "invalid-target.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="only 'Yes' and 'No'"):
        load_dataset(path)
