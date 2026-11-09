"""
Unit tests for Excel and CSV batch spreadsheet ingestion and last transaction simulation.
"""

import io
import pandas as pd
import pytest
from app.config import FEATURE_COLUMNS, AMOUNT_COLUMN, TIME_COLUMN
from app.decision_engine import FraudDecisionEngine
from app.batch_processor import generate_sample_template, process_transaction_file, export_scored_data
from ml.predictor import get_predictor


def test_generate_sample_template_xlsx():
    """Verify generated Excel template can be read back by pandas and contains proper columns."""
    excel_bytes = generate_sample_template(simulator=None, file_format="xlsx")
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0

    # Read back with openpyxl
    df = pd.read_excel(io.BytesIO(excel_bytes), engine="openpyxl")
    assert AMOUNT_COLUMN in df.columns
    assert TIME_COLUMN in df.columns
    for col in FEATURE_COLUMNS:
        assert col in df.columns
    assert len(df) == 10


def test_generate_sample_template_csv():
    """Verify generated CSV template contains proper headers and row count."""
    csv_bytes = generate_sample_template(simulator=None, file_format="csv")
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 0

    df = pd.read_csv(io.BytesIO(csv_bytes))
    assert AMOUNT_COLUMN in df.columns
    assert TIME_COLUMN in df.columns
    assert len(df) == 10


def test_process_excel_and_simulate_last_transaction():
    """Test full batch processing and last transaction simulation lifecycle from an Excel file."""
    excel_bytes = generate_sample_template(simulator=None, file_format="xlsx")
    predictor = get_predictor()
    engine = FraudDecisionEngine()

    scored_df, metrics, last_txn = process_transaction_file(
        io.BytesIO(excel_bytes), predictor, engine
    )

    # 1. Output verification
    assert len(scored_df) == 10
    assert "Transaction ID" in scored_df.columns
    assert "Fraud Probability" in scored_df.columns
    assert "Risk Level" in scored_df.columns
    assert "Decision" in scored_df.columns

    # 2. Metrics verification
    assert metrics["total_processed"] == 10
    assert metrics["allow_count"] + metrics["review_count"] + metrics["block_count"] == 10
    assert 0.0 <= metrics["max_fraud_prob"] <= 1.0

    # 3. Last transaction simulation verification
    assert last_txn is not None
    assert last_txn["transaction_id"] == "TXN-000010"
    assert "fraud_probability" in last_txn
    assert last_txn["risk_level"] in ("LOW", "MEDIUM", "HIGH")
    assert last_txn["decision"] in ("ALLOW", "REVIEW", "BLOCK")
    assert "explanation" in last_txn
    assert "action" in last_txn


def test_export_scored_data():
    """Verify scored data can be exported back to Excel and CSV bytes."""
    csv_bytes = generate_sample_template(simulator=None, file_format="csv")
    predictor = get_predictor()
    engine = FraudDecisionEngine()

    scored_df, _, _ = process_transaction_file(io.BytesIO(csv_bytes), predictor, engine)
    
    excel_out = export_scored_data(scored_df, file_format="xlsx")
    assert isinstance(excel_out, bytes)
    assert len(excel_out) > 0

    csv_out = export_scored_data(scored_df, file_format="csv")
    assert isinstance(csv_out, bytes)
    assert len(csv_out) > 0
