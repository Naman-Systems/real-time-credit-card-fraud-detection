"""
Transaction Replay Simulator Module.
Streams historical held-out test transactions sequentially into the Fraud Decision Engine
and In-Memory Store without exposing ground truth labels to the model or decision engine.
"""

from pathlib import Path
import sys
from typing import Dict, Any, Optional, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TIME_COLUMN,
    AMOUNT_COLUMN,
    PCA_FEATURES,
    DEFAULT_SIMULATION_INTERVAL_SEC,
)
from ml.preprocessing import load_dataset, chronological_split, validate_inference_features
from ml.predictor import FraudPredictor, get_predictor
from app.decision_engine import FraudDecisionEngine
from app.transaction_store import TransactionStore


class TransactionSimulator:
    """
    Simulates real-time transaction streaming by replaying held-out historical data.
    """

    def __init__(
        self,
        predictor: Optional[FraudPredictor] = None,
        decision_engine: Optional[FraudDecisionEngine] = None,
        store: Optional[TransactionStore] = None,
        test_data_path: Optional[Path] = None,
    ):
        self.predictor = predictor or get_predictor()
        self.decision_engine = decision_engine or FraudDecisionEngine()
        self.store = store or TransactionStore()
        
        self.test_data_path = test_data_path or (ARTIFACTS_DIR / "held_out_test.csv")
        self._test_df: Optional[pd.DataFrame] = None
        self._current_index: int = 0
        self._is_running: bool = False
        self._is_paused: bool = False
        self.interval_sec: float = DEFAULT_SIMULATION_INTERVAL_SEC
        
        self._load_held_out_data()

    def _load_held_out_data(self) -> None:
        """
        Load held-out test data. Checks uncompressed CSV, compressed gzip CSV,
        regenerates from raw dataset, or falls back to synthetic generation.
        """
        # 1. Try uncompressed held_out_test.csv
        if self.test_data_path.is_file():
            try:
                self._test_df = pd.read_csv(self.test_data_path)
                return
            except Exception:
                pass

        # 2. Try compressed held_out_test.csv.gz (tracked in git for cloud deployments)
        gz_path = self.test_data_path.with_suffix(".csv.gz")
        if gz_path.is_file():
            try:
                self._test_df = pd.read_csv(gz_path)
                return
            except Exception:
                pass

        # 3. Try rebuilding chronological split from raw dataset
        try:
            df = load_dataset()
            _, _, test_df = chronological_split(df)
            self._test_df = test_df
            self.test_data_path.parent.mkdir(parents=True, exist_ok=True)
            self._test_df.to_csv(self.test_data_path, index=False)
            return
        except Exception:
            pass

        # 4. Resilient Fallback: synthesize realistic streaming transactions
        self._test_df = self._generate_fallback_stream(n_samples=500)

    def _generate_fallback_stream(self, n_samples: int = 500) -> pd.DataFrame:
        """Generate realistic synthetic transactions for cloud deployments without raw datasets."""
        import numpy as np
        np.random.seed(42)
        rows = []
        for i in range(n_samples):
            is_fraud = (i % 25 == 0) and (i > 0)
            row = {f: float(np.random.normal(0, 1)) for f in PCA_FEATURES}
            row[AMOUNT_COLUMN] = round(float(np.random.exponential(scale=65.0) + 5.0), 2)
            row[TIME_COLUMN] = float(i * 120)
            if is_fraud:
                row["V14"] = float(np.random.uniform(-10.0, -5.0))
                row["V17"] = float(np.random.uniform(-8.0, -4.0))
                row["V12"] = float(np.random.uniform(-7.0, -3.5))
                row["V10"] = float(np.random.uniform(-6.0, -3.0))
                row["V4"] = float(np.random.uniform(2.5, 6.0))
                row["V11"] = float(np.random.uniform(2.0, 5.0))
                row[TARGET_COLUMN] = 1
            else:
                row[TARGET_COLUMN] = 0
            rows.append(row)
        return pd.DataFrame(rows)

    @property
    def total_stream_length(self) -> int:
        return len(self._test_df) if self._test_df is not None else 0

    @property
    def current_index(self) -> int:
        return self._current_index

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def is_finished(self) -> bool:
        return self._current_index >= self.total_stream_length

    def start(self) -> None:
        self._is_running = True
        self._is_paused = False

    def pause(self) -> None:
        self._is_paused = True

    def resume(self) -> None:
        self._is_paused = False
        self._is_running = True

    def stop(self) -> None:
        self._is_running = False
        self._is_paused = False

    def reset(self) -> None:
        self._is_running = False
        self._is_paused = False
        self._current_index = 0
        self.store.clear()

    def step(self) -> Optional[Dict[str, Any]]:
        """
        Replay a single held-out transaction through the end-to-end detection pipeline.
        
        Returns:
            The newly processed transaction record or None if at end-of-stream.
        """
        if self.is_finished:
            self.stop()
            return None

        # Extract current raw row
        row = self._test_df.iloc[self._current_index]
        self._current_index += 1

        # Extract true label separately for offline benchmark tracking only
        true_label = int(row[TARGET_COLUMN]) if TARGET_COLUMN in row else None

        # Build feature series strictly without Class
        feature_data = {col: float(row[col]) for col in FEATURE_COLUMNS}
        txn_df = pd.DataFrame([feature_data])

        # 1. Run inference pipeline
        fraud_prob = self.predictor.predict_probability(txn_df)

        # 2. Evaluate operational decision
        amount = float(row[AMOUNT_COLUMN])
        decision_result = self.decision_engine.evaluate(fraud_prob, amount=amount)

        # 3. Generate sequential ID
        txn_id = f"TXN-{self._current_index:06d}"
        elapsed = float(row[TIME_COLUMN])

        # 4. Store in memory with offline ground truth label
        record = self.store.add_transaction(
            transaction_id=txn_id,
            amount=amount,
            fraud_probability=decision_result["fraud_probability"],
            risk_level=decision_result["risk_level"],
            decision=decision_result["decision"],
            elapsed_time=elapsed,
            true_label=true_label,
            metadata={
                "explanation": decision_result["explanation"],
                "action": decision_result["action"],
            }
        )

        return record

    def step_batch(self, batch_size: int = 10) -> int:
        """
        Process a batch of simulated transactions.
        
        Args:
            batch_size: Number of transactions to process.
            
        Returns:
            Number of transactions actually processed.
        """
        count = 0
        for _ in range(batch_size):
            if self.is_finished:
                break
            res = self.step()
            if res is not None:
                count += 1
        return count
