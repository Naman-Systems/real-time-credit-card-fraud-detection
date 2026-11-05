"""
Unit tests for In-Memory Transaction and Alert Store.
"""

import pytest
from app.transaction_store import TransactionStore


def test_transaction_store_add_and_retrieve():
    store = TransactionStore(max_transactions=10, max_alerts=10)
    record = store.add_transaction(
        transaction_id="TXN-001",
        amount=150.0,
        fraud_probability=0.04,
        risk_level="LOW",
        decision="ALLOW"
    )
    assert record["transaction_id"] == "TXN-001"
    assert len(store.get_recent_transactions()) == 1


def test_transaction_store_duplicate_prevention():
    store = TransactionStore()
    store.add_transaction(
        transaction_id="TXN-DUP",
        amount=50.0,
        fraud_probability=0.1,
        risk_level="LOW",
        decision="ALLOW"
    )
    with pytest.raises(ValueError, match="Duplicate transaction ID"):
        store.add_transaction(
            transaction_id="TXN-DUP",
            amount=50.0,
            fraud_probability=0.1,
            risk_level="LOW",
            decision="ALLOW"
        )


def test_alert_generation_and_review():
    store = TransactionStore()
    # LOW risk should NOT generate alert
    store.add_transaction("TXN-1", 50.0, 0.05, "LOW", "ALLOW")
    assert len(store.get_alerts()) == 0

    # HIGH risk SHOULD generate alert
    store.add_transaction("TXN-2", 800.0, 0.88, "HIGH", "BLOCK")
    alerts = store.get_alerts()
    assert len(alerts) == 1
    assert alerts[0]["transaction_id"] == "TXN-2"
    assert alerts[0]["review_status"] == "PENDING"

    # Update review status
    updated = store.update_alert_status("TXN-2", "REVIEWED", notes="Confirmed fraud chargeback")
    assert updated is True
    assert store.get_alerts()[0]["review_status"] == "REVIEWED"


def test_store_buffer_capacity_eviction():
    store = TransactionStore(max_transactions=5, max_alerts=5)
    for i in range(10):
        store.add_transaction(f"TXN-{i}", 10.0 + i, 0.01, "LOW", "ALLOW")

    # Should only retain last 5 transactions
    recent = store.get_recent_transactions(limit=10)
    assert len(recent) == 5
    # Most recent first
    assert recent[0]["transaction_id"] == "TXN-9"
    assert recent[-1]["transaction_id"] == "TXN-5"


def test_store_summary_statistics():
    store = TransactionStore()
    store.add_transaction("TXN-A", 100.0, 0.02, "LOW", "ALLOW", true_label=0)
    store.add_transaction("TXN-B", 200.0, 0.35, "MEDIUM", "REVIEW", true_label=0)
    store.add_transaction("TXN-C", 500.0, 0.95, "HIGH", "BLOCK", true_label=1)

    stats = store.get_summary_statistics()
    assert stats["total_processed"] == 3
    assert stats["total_amount"] == 800.0
    assert stats["allow_count"] == 1
    assert stats["review_count"] == 1
    assert stats["block_count"] == 1
    assert stats["total_alerts"] == 2
    assert stats["simulation_accuracy"] == 1.0  # TXN-C predicted fraud & true label=1; TXN-A/B predicted legit & true label=0
