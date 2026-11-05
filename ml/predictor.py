"""
Production Predictor Module for Real-Time Fraud Detection.
Loads fitted Scikit-learn pipeline, enforces feature validation, prevents target leakage,
and computes fraud probabilities.
"""

from pathlib import Path
import sys
from typing import Dict, Any, Union, List
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import MODEL_PATH, FEATURE_COLUMNS, TARGET_COLUMN
from ml.preprocessing import validate_inference_features


class FraudPredictor:
    """
    Production-ready wrapper for the fitted fraud detection pipeline.
    """

    def __init__(self, model_path: Union[str, Path] = MODEL_PATH):
        self.model_path = Path(model_path)
        self.pipeline = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        """
        Load the persisted Joblib model artifact.
        """
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Trained model artifact not found at '{self.model_path}'.\n"
                "Please run `python ml/train.py` to train and generate the model pipeline."
            )
        try:
            self.pipeline = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model pipeline from '{self.model_path}': {e}") from e

    def predict_probability(self, transaction: Union[Dict[str, Any], pd.DataFrame, pd.Series]) -> float:
        """
        Compute the fraud probability for a single transaction.
        
        Args:
            transaction: Dictionary, Series, or 1-row DataFrame containing the 30 transaction features.
            
        Returns:
            Float probability in [0.0, 1.0].
            
        Raises:
            ValueError: If missing features or invalid input.
        """
        if isinstance(transaction, dict):
            df = pd.DataFrame([transaction])
        elif isinstance(transaction, pd.Series):
            df = pd.DataFrame([transaction.to_dict()])
        elif isinstance(transaction, pd.DataFrame):
            df = transaction.copy()
        else:
            raise TypeError(f"Unsupported transaction type: {type(transaction)}. Expected dict, Series, or DataFrame.")

        validated_df = validate_inference_features(df)
        
        # Verify no NaN values
        if validated_df.isnull().any().any():
            raise ValueError(f"Input contains NaN values in columns: {validated_df.columns[validated_df.isnull().any()].tolist()}")

        prob = float(self.pipeline.predict_proba(validated_df)[0, 1])
        return np.clip(prob, 0.0, 1.0)

    def predict_batch_probabilities(self, transactions_df: pd.DataFrame) -> np.ndarray:
        """
        Compute fraud probabilities for a batch of transactions.
        
        Args:
            transactions_df: DataFrame containing multiple transaction rows.
            
        Returns:
            1D numpy array of probabilities.
        """
        validated_df = validate_inference_features(transactions_df)
        if validated_df.isnull().any().any():
            raise ValueError("Input batch contains NaN values")

        probs = self.pipeline.predict_proba(validated_df)[:, 1]
        return np.clip(probs, 0.0, 1.0)


# Singleton instance helper
_predictor_instance = None


def get_predictor(model_path: Union[str, Path] = MODEL_PATH) -> FraudPredictor:
    """
    Get or initialize the FraudPredictor singleton instance.
    """
    global _predictor_instance
    if _predictor_instance is None or _predictor_instance.model_path != Path(model_path):
        _predictor_instance = FraudPredictor(model_path)
    return _predictor_instance
