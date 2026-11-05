"""
Fraud Risk Decision Engine Module.
Transforms machine learning fraud probabilities into operational risk levels
(LOW, MEDIUM, HIGH) and actionable business decisions (ALLOW, REVIEW, BLOCK).
"""

from typing import Dict, Any, Optional
import math
from app.config import (
    DEFAULT_LOW_RISK_THRESHOLD,
    DEFAULT_HIGH_RISK_THRESHOLD,
)


class FraudDecisionEngine:
    """
    Evaluates raw fraud probabilities against configurable operational risk bands.
    
    Operational Bands:
      - [0.0, low_threshold): LOW Risk -> ALLOW
      - [low_threshold, high_threshold): MEDIUM Risk -> REVIEW
      - [high_threshold, 1.0]: HIGH Risk -> BLOCK
    """

    def __init__(
        self,
        low_risk_threshold: float = DEFAULT_LOW_RISK_THRESHOLD,
        high_risk_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
    ):
        self.set_thresholds(low_risk_threshold, high_risk_threshold)

    def set_thresholds(self, low_threshold: float, high_threshold: float) -> None:
        """
        Update the operational decision risk thresholds.
        
        Args:
            low_threshold: Upper bound for LOW risk (ALLOW).
            high_threshold: Lower bound for HIGH risk (BLOCK).
            
        Raises:
            ValueError: If thresholds are invalid or out of order.
        """
        if not (0.0 <= low_threshold < high_threshold <= 1.0):
            raise ValueError(
                f"Invalid thresholds: low={low_threshold}, high={high_threshold}. "
                "Must satisfy 0.0 <= low < high <= 1.0."
            )
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)

    def evaluate(self, fraud_probability: float, amount: Optional[float] = None) -> Dict[str, Any]:
        """
        Evaluate a transaction fraud probability and render an operational decision.
        
        Args:
            fraud_probability: Model score in [0.0, 1.0].
            amount: Optional transaction amount in dollars.
            
        Returns:
            Dictionary with risk_level, decision, fraud_probability, explanation, and recommendation.
            
        Raises:
            ValueError: If probability is NaN, infinite, or outside [0.0, 1.0].
        """
        if fraud_probability is None or math.isnan(fraud_probability) or math.isinf(fraud_probability):
            raise ValueError(f"Invalid fraud probability: {fraud_probability}")

        if not (0.0 <= fraud_probability <= 1.0):
            raise ValueError(f"Fraud probability must be within [0.0, 1.0], got {fraud_probability}")

        prob = round(float(fraud_probability), 4)

        if prob < self.low_threshold:
            risk_level = "LOW"
            decision = "ALLOW"
            explanation = (
                f"Fraud probability ({prob:.2%}) is below the review threshold ({self.low_threshold:.0%}). "
                "Transaction behavior aligns closely with historical legitimate patterns."
            )
            action = "Transaction cleared for immediate processing."
        elif prob < self.high_threshold:
            risk_level = "MEDIUM"
            decision = "REVIEW"
            explanation = (
                f"Fraud probability ({prob:.2%}) falls into the intermediate risk band "
                f"[{self.low_threshold:.0%}, {self.high_threshold:.0%}). "
                "Transaction exhibits anomalous PCA or amount characteristics requiring manual inspection or step-up MFA."
            )
            action = "Hold funds temporarily and route to Risk Operations queue."
        else:
            risk_level = "HIGH"
            decision = "BLOCK"
            explanation = (
                f"Fraud probability ({prob:.2%}) exceeds the critical threshold ({self.high_threshold:.0%}). "
                "Transaction displays high statistical concordance with confirmed fraudulent patterns."
            )
            action = "Decline authorization and notify cardholder immediately."

        # High-value transaction contextual note
        if amount is not None and amount > 1000.0 and decision == "ALLOW":
            explanation += f" Note: Transaction amount (${amount:,.2f}) is substantial; secondary rule check advised."

        return {
            "fraud_probability": prob,
            "risk_level": risk_level,
            "decision": decision,
            "low_threshold": self.low_threshold,
            "high_threshold": self.high_threshold,
            "explanation": explanation,
            "action": action,
            "disclaimer": (
                "Notice: Model scores reflect statistical similarity to historical fraudulent activity "
                "and are probabilistic risk estimations rather than legal proof of fraud."
            ),
        }
