"""
Batch Processing and Spreadsheet Ingestion Engine.
Handles reading and parsing of Excel (.xlsx, .xls) and CSV files,
enforces feature validation and zero-leakage constraints, computes batch fraud
probabilities, renders operational decisions, and extracts the simulated last transaction.
"""

import io
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd

from app.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    AMOUNT_COLUMN,
    PCA_FEATURES,
    ARTIFACTS_DIR,
)
from ml.predictor import FraudPredictor, get_predictor
from app.decision_engine import FraudDecisionEngine


def generate_sample_template(simulator=None, file_format: str = "xlsx") -> bytes:
    """
    Generate a sample transaction spreadsheet preloaded with realistic test transactions.
    Uses 5 legitimate and 5 fraudulent historical/realistic transactions.
    
    Args:
        simulator: Optional TransactionSimulator instance with loaded test data.
        file_format: 'xlsx' or 'csv'.
        
    Returns:
        Raw bytes of the generated spreadsheet.
    """
    sample_df = None
    if simulator is not None and getattr(simulator, "_test_df", None) is not None:
        test_df = simulator._test_df
        if TARGET_COLUMN in test_df.columns:
            legit = test_df[test_df[TARGET_COLUMN] == 0].head(5)
            fraud = test_df[test_df[TARGET_COLUMN] == 1].head(5)
            sample_df = pd.concat([legit, fraud], ignore_index=True)

    if sample_df is None or sample_df.empty:
        # Check if held-out test file is directly readable
        held_out_path = ARTIFACTS_DIR / "held_out_test.csv"
        held_out_gz = ARTIFACTS_DIR / "held_out_test.csv.gz"
        for p in (held_out_path, held_out_gz):
            if p.is_file():
                try:
                    loaded = pd.read_csv(p)
                    if TARGET_COLUMN in loaded.columns:
                        legit = loaded[loaded[TARGET_COLUMN] == 0].head(5)
                        fraud = loaded[loaded[TARGET_COLUMN] == 1].head(5)
                        sample_df = pd.concat([legit, fraud], ignore_index=True)
                        break
                except Exception:
                    pass

    if sample_df is None or sample_df.empty:
        # Resilient fallback: realistic transactions
        np.random.seed(42)
        rows = []
        for i in range(10):
            row = {col: 0.0 for col in FEATURE_COLUMNS}
            row[AMOUNT_COLUMN] = round(float((i + 1) * 35.50 + np.random.uniform(5, 20)), 2)
            row[TIME_COLUMN] = float(i * 120 + 300)
            if i >= 5:  # Fraudulent profile
                row["V14"] = -8.2
                row["V17"] = -6.5
                row["V12"] = -5.0
                row["V10"] = -4.3
                row["V4"] = 3.8
                row["V11"] = 3.2
                row[TARGET_COLUMN] = 1
            else:
                row[TARGET_COLUMN] = 0
            rows.append(row)
        sample_df = pd.DataFrame(rows)

    # Exclude target column from download template so user gets clean inference template
    columns_to_export = [c for c in FEATURE_COLUMNS if c in sample_df.columns]
    export_df = sample_df[columns_to_export].copy()

    if file_format.lower() in ("xlsx", "excel"):
        buffer = io.BytesIO()
        export_df.to_excel(buffer, index=False, engine="openpyxl")
        return buffer.getvalue()
    else:
        return export_df.to_csv(index=False).encode("utf-8")


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Read an uploaded Streamlit file (Excel or CSV) into a pandas DataFrame.
    """
    filename = getattr(uploaded_file, "name", "").lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        # Attempt CSV first, then Excel fallback
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_excel(uploaded_file, engine="openpyxl")


def process_transaction_file(
    file_or_df: Union[Any, pd.DataFrame],
    predictor: FraudPredictor,
    decision_engine: FraudDecisionEngine,
    store: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Parse uploaded transactions, execute batch inference, evaluate operational risk decisions,
    compute aggregated summary metrics, optionally register alerts in the TransactionStore,
    and extract the simulated last processed transaction.
    
    Args:
        file_or_df: Uploaded file object, bytes buffer, or existing DataFrame.
        predictor: Production FraudPredictor instance.
        decision_engine: Configured FraudDecisionEngine instance.
        store: Optional TransactionStore instance to populate with transactions and alerts.
        
    Returns:
        Tuple of (scored_df, metrics_dict, last_processed_transaction_dict).
    """
    if isinstance(file_or_df, pd.DataFrame):
        df = file_or_df.copy()
    else:
        df = read_uploaded_file(file_or_df)

    if df.empty:
        raise ValueError("Uploaded file contains no transaction records.")

    # 1. Normalize Column Names (handle casing differences)
    col_mapping = {}
    lower_map = {c.lower(): c for c in df.columns}
    
    if "time" in lower_map:
        col_mapping[lower_map["time"]] = TIME_COLUMN
    if "amount" in lower_map:
        col_mapping[lower_map["amount"]] = AMOUNT_COLUMN
    for i in range(1, 29):
        v_key = f"v{i}"
        if v_key in lower_map:
            col_mapping[lower_map[v_key]] = f"V{i}"

    if col_mapping:
        df = df.rename(columns=col_mapping)

    # 2. Prevent target leakage (drop Class if present in uploaded file)
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    # 3. Ensure essential columns exist; gracefully impute defaults if missing
    if TIME_COLUMN not in df.columns:
        df[TIME_COLUMN] = np.arange(len(df)) * 60.0

    if AMOUNT_COLUMN not in df.columns:
        raise ValueError("Spreadsheet must include an 'Amount' column specifying transaction values.")

    # Convert Amount & Time to float and clean any currency formatting
    if df[AMOUNT_COLUMN].dtype == object:
        df[AMOUNT_COLUMN] = (
            df[AMOUNT_COLUMN]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
            .astype(float)
        )

    # Ensure PCA features exist (impute 0.0 if not provided)
    for feat in PCA_FEATURES:
        if feat not in df.columns:
            df[feat] = 0.0

    # Fill any isolated NaNs with zeros
    features_df = df[FEATURE_COLUMNS].fillna(0.0).copy()

    # 4. Predict fraud probabilities
    probabilities = predictor.predict_batch_probabilities(features_df)

    # 5. Evaluate Operational Decisions
    scored_records = []
    allow_count = 0
    review_count = 0
    block_count = 0

    if store is not None:
        store.clear()

    for idx, (original_idx, row) in enumerate(features_df.iterrows()):
        prob = float(probabilities[idx])
        amount = float(row[AMOUNT_COLUMN])
        elapsed = float(row[TIME_COLUMN])

        decision_info = decision_engine.evaluate(prob, amount=amount)
        risk = decision_info["risk_level"]
        decision = decision_info["decision"]

        if decision == "ALLOW":
            allow_count += 1
        elif decision == "REVIEW":
            review_count += 1
        else:
            block_count += 1

        txn_id = f"TXN-{idx + 1:06d}"
        record = {
            "Transaction ID": txn_id,
            "Time (s)": round(elapsed, 1),
            "Amount ($)": round(amount, 2),
            "Fraud Probability": prob,
            "Risk Level": risk,
            "Decision": decision,
            "Explanation": decision_info["explanation"],
            "Action": decision_info["action"],
        }
        scored_records.append(record)

        if store is not None:
            store.add_transaction(
                transaction_id=txn_id,
                amount=amount,
                fraud_probability=prob,
                risk_level=risk,
                decision=decision,
                elapsed_time=elapsed,
                metadata={
                    "explanation": decision_info["explanation"],
                    "action": decision_info["action"],
                },
            )

    scored_df = pd.DataFrame(scored_records)
    total_count = len(scored_df)

    # 6. Aggregate Summary Metrics
    metrics = {
        "total_processed": total_count,
        "allow_count": allow_count,
        "review_count": review_count,
        "block_count": block_count,
        "allow_rate": (allow_count / total_count * 100.0) if total_count > 0 else 0.0,
        "review_rate": (review_count / total_count * 100.0) if total_count > 0 else 0.0,
        "block_rate": (block_count / total_count * 100.0) if total_count > 0 else 0.0,
        "max_fraud_prob": float(np.max(probabilities)) if total_count > 0 else 0.0,
        "avg_fraud_prob": float(np.mean(probabilities)) if total_count > 0 else 0.0,
    }

    # 7. Extract Simulated Last Processed Transaction
    last_row_features = features_df.iloc[-1].to_dict()
    last_record = scored_records[-1].copy()
    last_processed_transaction = {
        "transaction_id": last_record["Transaction ID"],
        "elapsed_time": last_record["Time (s)"],
        "amount": last_record["Amount ($)"],
        "fraud_probability": last_record["Fraud Probability"],
        "risk_level": last_record["Risk Level"],
        "decision": last_record["Decision"],
        "explanation": last_record["Explanation"],
        "action": last_record["Action"],
        "features": last_row_features,
    }

    return scored_df, metrics, last_processed_transaction


def export_scored_data(scored_df: pd.DataFrame, file_format: str = "xlsx") -> bytes:
    """
    Export the scored transaction dataframe into downloadable Excel or CSV bytes.
    """
    if file_format.lower() in ("xlsx", "excel"):
        buf = io.BytesIO()
        scored_df.to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue()
    else:
        return scored_df.to_csv(index=False).encode("utf-8")
