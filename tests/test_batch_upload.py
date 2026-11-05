"""
Unit tests for Excel and batch spreadsheet ingestion in the Live Monitoring module.
"""

import io
import pandas as pd
import pytest
from app.config import FEATURE_COLUMNS, AMOUNT_COLUMN, TIME_COLUMN
from app.decision_engine import FraudDecisionEngine
from app.pages.live_monitoring import generate_sample_template
from ml.predictor import get_predictor


def test_generate_sample_template_xlsx():
    """Verify generated Excel template can be read back by pandas and contains proper columns."""
    excel_bytes = generate_sample_template(simulator=None, file_format="xlsx")
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0

    # Read back with openpyxl
    df = pd.read_excel(io.BytesIO(excel_bytes))
    assert AMOUNT_COLUMN in df.columns
    assert TIME_COLUMN in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert len(df) == 10


def test_generate_sample_template_csv():
    """Verify generated CSV template contains proper headers."""
    csv_bytes = generate_sample_template(simulator=None, file_format="csv")
    assert isinstance(csv_bytes, bytes)
    df = pd.read_csv(io.BytesIO(csv_bytes))
    assert AMOUNT_COLUMN in df.columns
    assert len(df) == 10


def test_batch_prediction_from_excel_data():
    """Test full batch prediction lifecycle on an ingested dataframe."""
    excel_bytes = generate_sample_template(simulator=None, file_format="xlsx")
    df = pd.read_excel(io.BytesIO(excel_bytes))

    predictor = get_predictor()
    engine = FraudDecisionEngine()

    features_df = df[FEATURE_COLUMNS]
    probs = predictor.predict_batch_probabilities(features_df)

    assert len(probs) == len(df)
    assert all(0.0 <= p <= 1.0 for p in probs)

    for i, p in enumerate(probs):
        amt = float(df.iloc[i][AMOUNT_COLUMN])
        res = engine.evaluate(p, amount=amt)
        assert res["decision"] in ("ALLOW", "REVIEW", "BLOCK")
        assert res["risk_level"] in ("LOW", "MEDIUM", "HIGH")
