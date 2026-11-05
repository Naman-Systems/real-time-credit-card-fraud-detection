"""
Unit tests for model pipeline persistence, prediction, and probability outputs.
"""

import pytest
import numpy as np
import pandas as pd
from app.config import MODEL_PATH, FEATURE_COLUMNS, METADATA_PATH, EVALUATION_METRICS_PATH
from ml.predictor import FraudPredictor, get_predictor


@pytest.fixture(scope="module")
def predictor():
    return get_predictor()


def test_model_artifact_exists():
    assert MODEL_PATH.is_file(), f"Model file missing: {MODEL_PATH}"
    assert METADATA_PATH.is_file(), f"Metadata file missing: {METADATA_PATH}"
    assert EVALUATION_METRICS_PATH.is_file(), f"Metrics file missing: {EVALUATION_METRICS_PATH}"


def test_single_prediction_probability_range(predictor):
    sample = {col: 0.0 for col in FEATURE_COLUMNS}
    sample["Amount"] = 120.50
    sample["Time"] = 3600.0

    prob = predictor.predict_probability(sample)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_batch_prediction(predictor):
    np.random.seed(42)
    rows = []
    for _ in range(5):
        row = {f"V{i}": np.random.normal(0, 1) for i in range(1, 29)}
        row["Time"] = np.random.uniform(10, 5000)
        row["Amount"] = np.random.uniform(5, 500)
        rows.append(row)
    df = pd.DataFrame(rows)

    probs = predictor.predict_batch_probabilities(df)
    assert len(probs) == 5
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_prediction_rejects_missing_features(predictor):
    incomplete = {"V1": 0.5, "Amount": 10.0}
    with pytest.raises(ValueError, match="Missing required inference features"):
        predictor.predict_probability(incomplete)
