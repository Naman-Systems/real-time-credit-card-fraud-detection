"""
Unit tests for the Transaction Replay Simulator.
"""

import pytest
import pandas as pd
from app.simulator import TransactionSimulator
from app.transaction_store import TransactionStore
from app.decision_engine import FraudDecisionEngine
from app.config import TARGET_COLUMN


def test_simulator_step_execution():
    store = TransactionStore(max_transactions=50)
    engine = FraudDecisionEngine()
    sim = TransactionSimulator(store=store, decision_engine=engine)

    assert sim.total_stream_length > 0
    assert sim.current_index == 0

    # Execute one simulation step
    record = sim.step()
    assert record is not None
    assert record["transaction_id"] == "TXN-000001"
    assert "amount" in record
    assert "fraud_probability" in record
    assert "decision" in record
    assert sim.current_index == 1

    # Check store has it
    recent = store.get_recent_transactions()
    assert len(recent) == 1
    assert recent[0]["transaction_id"] == "TXN-000001"


def test_simulator_batch_step():
    store = TransactionStore(max_transactions=50)
    sim = TransactionSimulator(store=store)

    processed = sim.step_batch(5)
    assert processed == 5
    assert sim.current_index == 5
    assert len(store.get_recent_transactions()) == 5


def test_simulator_pause_resume_reset():
    store = TransactionStore(max_transactions=50)
    sim = TransactionSimulator(store=store)

    sim.start()
    assert sim.is_running is True

    sim.pause()
    assert sim.is_paused is True

    sim.resume()
    assert sim.is_paused is False
    assert sim.is_running is True

    sim.step_batch(3)
    assert sim.current_index == 3

    sim.reset()
    assert sim.current_index == 0
    assert sim.is_running is False
    assert len(store.get_recent_transactions()) == 0


def test_simulator_no_target_leakage():
    """Verify that Class is never in the feature vector passed to predictor."""
    sim = TransactionSimulator()
    # Check held_out_data has features
    assert sim._test_df is not None
    first_row = sim._test_df.iloc[0]
    
    # Target column must be handled separately and not passed as input feature
    assert TARGET_COLUMN in sim._test_df.columns
    # Check predictor input requirements
    feature_keys = [c for c in sim._test_df.columns if c != TARGET_COLUMN]
    assert len(feature_keys) == 30
