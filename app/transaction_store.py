"""
In-Memory Transaction and Alert Store Module.
Provides bounded in-memory buffer for real-time transaction processing,
fraud alerts, and aggregate KPI calculation using Python structures and Streamlit session state.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import deque
import pandas as pd

from app.config import (
    MAX_STORED_TRANSACTIONS,
    MAX_STORED_ALERTS,
)


class TransactionStore:
    """
    Thread-safe in-memory store for recent simulated and manual transactions.
    
    Attributes:
        max_transactions: Maximum number of transactions retained in memory (default 1000).
        max_alerts: Maximum number of alerts retained in memory (default 1000).
    """

    def __init__(
        self,
        max_transactions: int = MAX_STORED_TRANSACTIONS,
        max_alerts: int = MAX_STORED_ALERTS,
    ):
        self.max_transactions = max_transactions
        self.max_alerts = max_alerts
        self._transactions: deque = deque(maxlen=max_transactions)
        self._alerts: deque = deque(maxlen=max_alerts)
        self._transaction_ids: set = set()
        
        # Ground truth labels recorded purely for offline simulation evaluation
        # Never exposed to operational decision paths
        self._offline_ground_truth: Dict[str, int] = {}

    def add_transaction(
        self,
        transaction_id: str,
        amount: float,
        fraud_probability: float,
        risk_level: str,
        decision: str,
        timestamp: Optional[str] = None,
        elapsed_time: Optional[float] = None,
        true_label: Optional[int] = None,
        review_status: str = "PENDING",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a newly processed transaction.
        
        Args:
            transaction_id: Unique transaction reference.
            amount: Dollar amount.
            fraud_probability: Model score [0.0, 1.0].
            risk_level: 'LOW', 'MEDIUM', or 'HIGH'.
            decision: 'ALLOW', 'REVIEW', or 'BLOCK'.
            timestamp: Formatted timestamp string (defaults to UTC/local now).
            elapsed_time: Dataset elapsed seconds from reference.
            true_label: Optional ground truth Class (kept only for offline simulation accuracy).
            review_status: Initial review status ('PENDING', 'REVIEWED').
            metadata: Additional dictionary payload.
            
        Returns:
            The recorded transaction dictionary.
            
        Raises:
            ValueError: If transaction_id already exists.
        """
        if transaction_id in self._transaction_ids:
            raise ValueError(f"Duplicate transaction ID: {transaction_id}")

        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "elapsed_time": elapsed_time if elapsed_time is not None else 0.0,
            "amount": round(float(amount), 2),
            "fraud_probability": round(float(fraud_probability), 4),
            "risk_level": risk_level,
            "decision": decision,
            "review_status": review_status,
            "metadata": metadata or {},
        }

        # Handle capacity overflow in tracking set
        if len(self._transactions) == self.max_transactions:
            evicted = self._transactions[0]
            self._transaction_ids.discard(evicted["transaction_id"])
            self._offline_ground_truth.pop(evicted["transaction_id"], None)

        self._transactions.append(record)
        self._transaction_ids.add(transaction_id)

        # Store offline ground truth separately if available
        if true_label is not None:
            self._offline_ground_truth[transaction_id] = int(true_label)

        # If flagged as MEDIUM or HIGH risk, add to alerts queue
        if risk_level in ("MEDIUM", "HIGH"):
            alert_record = {
                "alert_id": f"ALT-{transaction_id}",
                "transaction_id": transaction_id,
                "timestamp": timestamp,
                "amount": record["amount"],
                "fraud_probability": record["fraud_probability"],
                "risk_level": risk_level,
                "decision": decision,
                "review_status": review_status,
                "reviewer_notes": "",
            }
            if len(self._alerts) == self.max_alerts:
                self._alerts.popleft()
            self._alerts.append(alert_record)

        return record

    def get_recent_transactions(
        self,
        limit: int = 50,
        risk_filter: Optional[str] = None,
        decision_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the most recent transactions, optionally filtered.
        """
        results = list(self._transactions)[::-1]  # Most recent first

        if risk_filter and risk_filter != "ALL":
            results = [r for r in results if r["risk_level"] == risk_filter]

        if decision_filter and decision_filter != "ALL":
            results = [r for r in results if r["decision"] == decision_filter]

        return results[:limit]

    def get_alerts(
        self,
        risk_filter: Optional[str] = None,
        status_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return alert queue sorted by risk and recency.
        """
        alerts = list(self._alerts)[::-1]

        if risk_filter and risk_filter != "ALL":
            alerts = [a for a in alerts if a["risk_level"] == risk_filter]

        if status_filter and status_filter != "ALL":
            alerts = [a for a in alerts if a["review_status"] == status_filter]

        return alerts

    def update_alert_status(self, transaction_id: str, new_status: str, notes: str = "") -> bool:
        """
        Update the review status of an alert and its underlying transaction.
        """
        found = False
        for alert in self._alerts:
            if alert["transaction_id"] == transaction_id:
                alert["review_status"] = new_status
                alert["reviewer_notes"] = notes
                found = True
                break

        for txn in self._transactions:
            if txn["transaction_id"] == transaction_id:
                txn["review_status"] = new_status
                break

        return found

    def clear(self) -> None:
        """
        Reset transaction and alert histories.
        """
        self._transactions.clear()
        self._alerts.clear()
        self._transaction_ids.clear()
        self._offline_ground_truth.clear()

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Calculate live streaming aggregate metrics.
        """
        total = len(self._transactions)
        if total == 0:
            return {
                "total_processed": 0,
                "total_amount": 0.0,
                "allow_count": 0,
                "review_count": 0,
                "block_count": 0,
                "low_risk_count": 0,
                "medium_risk_count": 0,
                "high_risk_count": 0,
                "fraud_suspicion_rate": 0.0,
                "avg_fraud_probability": 0.0,
                "avg_amount": 0.0,
                "total_alerts": 0,
                "pending_alerts": 0,
                "simulation_accuracy": None,
            }

        allow_count = sum(1 for t in self._transactions if t["decision"] == "ALLOW")
        review_count = sum(1 for t in self._transactions if t["decision"] == "REVIEW")
        block_count = sum(1 for t in self._transactions if t["decision"] == "BLOCK")

        low_count = sum(1 for t in self._transactions if t["risk_level"] == "LOW")
        med_count = sum(1 for t in self._transactions if t["risk_level"] == "MEDIUM")
        high_count = sum(1 for t in self._transactions if t["risk_level"] == "HIGH")

        total_amount = sum(t["amount"] for t in self._transactions)
        total_probs = sum(t["fraud_probability"] for t in self._transactions)

        pending_alerts = sum(1 for a in self._alerts if a["review_status"] == "PENDING")

        # Offline accuracy calculation if labels were provided
        labeled_count = len(self._offline_ground_truth)
        sim_acc = None
        if labeled_count > 0:
            correct = 0
            for t in self._transactions:
                tid = t["transaction_id"]
                if tid in self._offline_ground_truth:
                    actual = self._offline_ground_truth[tid]
                    predicted_fraud = 1 if t["decision"] == "BLOCK" or t["risk_level"] == "HIGH" else 0
                    if actual == predicted_fraud:
                        correct += 1
            sim_acc = round(correct / labeled_count, 4)

        return {
            "total_processed": total,
            "total_amount": round(total_amount, 2),
            "allow_count": allow_count,
            "review_count": review_count,
            "block_count": block_count,
            "low_risk_count": low_count,
            "medium_risk_count": med_count,
            "high_risk_count": high_count,
            "fraud_suspicion_rate": round(((med_count + high_count) / total) * 100, 2),
            "avg_fraud_probability": round(total_probs / total, 4),
            "avg_amount": round(total_amount / total, 2),
            "total_alerts": len(self._alerts),
            "pending_alerts": pending_alerts,
            "simulation_accuracy": sim_acc,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert transaction buffer to pandas DataFrame for plotting and exports.
        """
        if not self._transactions:
            return pd.DataFrame()
        return pd.DataFrame(list(self._transactions))
