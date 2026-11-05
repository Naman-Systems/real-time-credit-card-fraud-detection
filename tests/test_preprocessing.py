"""
Unit tests for data preprocessing and feature validation.
"""

import pytest
import numpy as np
import pandas as pd
from app.config import FEATURE_COLUMNS, TARGET_COLUMN, TIME_COLUMN, AMOUNT_COLUMN
from ml.preprocessing import (
    validate_dataset_schema,
    validate_inference_features,
    chronological_split,
    build_feature_preprocessor,
)


def create_synthetic_transaction_df(n_samples: int = 100) -> pd.DataFrame:
    """Helper fixture creating synthetic transaction DataFrame."""
    np.random.seed(42)
    data = {TIME_COLUMN: np.sort(np.random.uniform(0, 10000, n_samples))}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n_samples)
    data[AMOUNT_COLUMN] = np.random.exponential(50, n_samples)
    # Rare fraud labels (approx 2%)
    data[TARGET_COLUMN] = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    return pd.DataFrame(data)


def test_validate_dataset_schema_success():
    df = create_synthetic_transaction_df(20)
    # Should not raise
    validate_dataset_schema(df)


def test_validate_dataset_schema_missing_column():
    df = create_synthetic_transaction_df(20).drop(columns=["V10"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset_schema(df)


def test_validate_inference_features_removes_target():
    """Ensure target column 'Class' is stripped to prevent data leakage."""
    df = create_synthetic_transaction_df(10)
    assert TARGET_COLUMN in df.columns
    cleaned = validate_inference_features(df)
    assert TARGET_COLUMN not in cleaned.columns
    assert list(cleaned.columns) == FEATURE_COLUMNS


def test_validate_inference_features_missing_column():
    df = create_synthetic_transaction_df(10).drop(columns=["Amount"])
    with pytest.raises(ValueError, match="Missing required inference features"):
        validate_inference_features(df)


def test_chronological_split_ordering():
    df = create_synthetic_transaction_df(100)
    train_df, val_df, test_df = chronological_split(df, 0.7, 0.15, 0.15)
    
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Check chronological boundary
    assert train_df[TIME_COLUMN].max() <= val_df[TIME_COLUMN].min()
    assert val_df[TIME_COLUMN].max() <= test_df[TIME_COLUMN].min()


def test_feature_preprocessor_transformation():
    df = create_synthetic_transaction_df(50)
    X = validate_inference_features(df)
    preprocessor = build_feature_preprocessor()
    transformed = preprocessor.fit_transform(X)
    
    # 30 input features should yield 30 transformed output features
    assert transformed.shape == (50, 30)
    # Check no NaNs
    assert not np.isnan(transformed).any()
