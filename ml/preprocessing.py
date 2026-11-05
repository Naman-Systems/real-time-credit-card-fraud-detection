"""
Data Preprocessing and Pipeline Module for Credit Card Fraud Detection.
Handles dataset loading, validation, chronological splitting, feature scaling,
and data leakage prevention.
"""

from pathlib import Path
import sys
from typing import Tuple, List, Optional, Dict, Any
import logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler

from app.config import (
    DATASET_PATH,
    TARGET_COLUMN,
    TIME_COLUMN,
    AMOUNT_COLUMN,
    PCA_FEATURES,
    FEATURE_COLUMNS,
    ALL_DATASET_COLUMNS,
)

logger = logging.getLogger(__name__)


def load_dataset(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load historical transaction data from CSV.
    
    Args:
        filepath: Optional path to creditcard.csv. Defaults to config DATASET_PATH.
        
    Returns:
        pd.DataFrame containing historical transactions.
        
    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If required columns are missing from the dataset.
    """
    path = Path(filepath) if filepath else DATASET_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Please download the ULB Credit Card Fraud Detection dataset from Kaggle:\n"
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            f"and place 'creditcard.csv' into the '{path.parent}' directory."
        )

    logger.info("Loading dataset from %s ...", path)
    df = pd.read_csv(path)
    validate_dataset_schema(df)
    logger.info("Successfully loaded dataset with shape %s", df.shape)
    return df


def validate_dataset_schema(df: pd.DataFrame) -> None:
    """
    Verify that the DataFrame contains all expected columns and target column.
    
    Args:
        df: Input DataFrame to check.
        
    Raises:
        ValueError: If any required column is missing.
    """
    missing_cols = [col for col in ALL_DATASET_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")


def validate_inference_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and enforce column structure for model inference.
    Strictly prevents target leakage by stripping 'Class' if present,
    and guarantees exact feature ordering.
    
    Args:
        features_df: Input DataFrame with transaction features.
        
    Returns:
        pd.DataFrame with exactly FEATURE_COLUMNS in the correct order.
        
    Raises:
        ValueError: If any feature from FEATURE_COLUMNS is missing.
    """
    # Prevent leakage: remove Class if erroneously passed
    df = features_df.copy()
    if TARGET_COLUMN in df.columns:
        logger.warning("Target column '%s' detected in features. Removing to prevent data leakage.", TARGET_COLUMN)
        df = df.drop(columns=[TARGET_COLUMN])

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required inference features: {missing}")

    # Reorder columns strictly to match training order
    return df[FEATURE_COLUMNS]


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically based on elapsed Time to emulate production deployment.
    Does not use future transactions to train or validate earlier models.
    
    Args:
        df: Full transaction DataFrame.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation / threshold tuning (default 0.15).
        test_ratio: Fraction for final out-of-time evaluation & replay (default 0.15).
        
    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    # Sort strictly by Time to preserve chronological order
    df_sorted = df.sort_values(by=TIME_COLUMN).reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(train_ratio * n)
    val_end = int((train_ratio + val_ratio) * n)

    train_df = df_sorted.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df_sorted.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df_sorted.iloc[val_end:].copy().reset_index(drop=True)

    logger.info(
        "Chronological Split Summary:\n"
        "  - Train: %d rows (Fraud: %d, %.3f%%)\n"
        "  - Val:   %d rows (Fraud: %d, %.3f%%)\n"
        "  - Test:  %d rows (Fraud: %d, %.3f%%)",
        len(train_df), train_df[TARGET_COLUMN].sum(), train_df[TARGET_COLUMN].mean() * 100,
        len(val_df), val_df[TARGET_COLUMN].sum(), val_df[TARGET_COLUMN].mean() * 100,
        len(test_df), test_df[TARGET_COLUMN].sum(), test_df[TARGET_COLUMN].mean() * 100,
    )

    return train_df, val_df, test_df


def build_feature_preprocessor() -> ColumnTransformer:
    """
    Construct a Scikit-learn ColumnTransformer for feature scaling.
    Uses RobustScaler for 'Amount' and 'Time' because of severe outliers.
    PCA features V1-V28 are passed through unmodified because they are
    already orthogonal, mean-centered, and variance-scaled via PCA.
    
    Returns:
        ColumnTransformer instance.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("amount_scaler", RobustScaler(), [AMOUNT_COLUMN]),
            ("time_scaler", RobustScaler(), [TIME_COLUMN]),
            ("pca_passthrough", "passthrough", PCA_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics for EDA and Dataset Explorer.
    
    Args:
        df: Transaction DataFrame.
        
    Returns:
        Dictionary with dataset health and distribution stats.
    """
    total_count = len(df)
    fraud_count = int(df[TARGET_COLUMN].sum()) if TARGET_COLUMN in df.columns else 0
    legit_count = total_count - fraud_count
    fraud_pct = (fraud_count / total_count * 100.0) if total_count > 0 else 0.0

    return {
        "total_transactions": total_count,
        "legitimate_transactions": legit_count,
        "fraudulent_transactions": fraud_count,
        "fraud_percentage": fraud_pct,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "time_min": float(df[TIME_COLUMN].min()) if TIME_COLUMN in df.columns else 0.0,
        "time_max": float(df[TIME_COLUMN].max()) if TIME_COLUMN in df.columns else 0.0,
        "amount_mean": float(df[AMOUNT_COLUMN].mean()) if AMOUNT_COLUMN in df.columns else 0.0,
        "amount_median": float(df[AMOUNT_COLUMN].median()) if AMOUNT_COLUMN in df.columns else 0.0,
        "amount_max": float(df[AMOUNT_COLUMN].max()) if AMOUNT_COLUMN in df.columns else 0.0,
        "amount_min": float(df[AMOUNT_COLUMN].min()) if AMOUNT_COLUMN in df.columns else 0.0,
    }
