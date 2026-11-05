"""
Configuration module for the Real-Time Credit Card Fraud Detection System.
Defines system paths, feature definitions, decision thresholds, and UI display settings.
"""

from pathlib import Path
from typing import Dict, List, Any

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
DATASET_PATH = DATASET_DIR / "creditcard.csv"

ML_DIR = BASE_DIR / "ml"
ARTIFACTS_DIR = ML_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "fraud_pipeline.pkl"
METADATA_PATH = ARTIFACTS_DIR / "metadata.json"
EVALUATION_METRICS_PATH = ARTIFACTS_DIR / "evaluation_metrics.json"

# Cloud-safe bundled dataset assets (compressed, tracked in git)
HELD_OUT_TEST_GZ_PATH = ARTIFACTS_DIR / "held_out_test.csv.gz"
EDA_SAMPLE_GZ_PATH = ARTIFACTS_DIR / "eda_sample.csv.gz"

# Feature definitions
TARGET_COLUMN = "Class"
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]

# Exact feature ordering expected by model pipeline (30 features total)
FEATURE_COLUMNS: List[str] = [TIME_COLUMN] + PCA_FEATURES + [AMOUNT_COLUMN]

# Complete raw dataset columns (31 columns)
ALL_DATASET_COLUMNS: List[str] = FEATURE_COLUMNS + [TARGET_COLUMN]

# Default Decision Engine Risk Bands
# Risk bands define business operations:
# - Probability < LOW_RISK_THRESHOLD: LOW risk -> ALLOW
# - LOW_RISK_THRESHOLD <= Probability < HIGH_RISK_THRESHOLD: MEDIUM risk -> REVIEW
# - Probability >= HIGH_RISK_THRESHOLD: HIGH risk -> BLOCK
DEFAULT_LOW_RISK_THRESHOLD: float = 0.20
DEFAULT_HIGH_RISK_THRESHOLD: float = 0.70

# Optimal binary classification threshold (tuned on validation PR curve / F1)
DEFAULT_CLASSIFICATION_THRESHOLD: float = 0.50

# In-Memory Buffer Limits
MAX_STORED_TRANSACTIONS: int = 1000
MAX_STORED_ALERTS: int = 1000

# Simulator settings
DEFAULT_SIMULATION_INTERVAL_SEC: float = 1.0
MIN_SIMULATION_INTERVAL_SEC: float = 0.1
MAX_SIMULATION_INTERVAL_SEC: float = 5.0

# UI Theme & Status Colors
RISK_COLORS: Dict[str, str] = {
    "LOW": "#10B981",       # Emerald Green
    "MEDIUM": "#F59E0B",    # Amber
    "HIGH": "#EF4444"       # Rose Red
}

DECISION_COLORS: Dict[str, str] = {
    "ALLOW": "#10B981",
    "REVIEW": "#F59E0B",
    "BLOCK": "#EF4444"
}

REVIEW_STATUS_COLORS: Dict[str, str] = {
    "PENDING": "#F59E0B",
    "REVIEWED": "#10B981",
    "DISMISSED": "#6B7280"
}
