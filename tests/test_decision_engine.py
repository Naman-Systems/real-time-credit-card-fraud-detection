"""
Unit tests for the Fraud Decision Engine and risk band classifications.
"""

import pytest
from app.decision_engine import FraudDecisionEngine


def test_decision_engine_default_thresholds():
    engine = FraudDecisionEngine()
    assert engine.low_threshold == 0.20
    assert engine.high_threshold == 0.70

    # Low risk (ALLOW)
    res_low = engine.evaluate(0.05, amount=50.0)
    assert res_low["risk_level"] == "LOW"
    assert res_low["decision"] == "ALLOW"

    # Medium risk (REVIEW)
    res_med = engine.evaluate(0.45, amount=200.0)
    assert res_med["risk_level"] == "MEDIUM"
    assert res_med["decision"] == "REVIEW"

    # High risk (BLOCK)
    res_high = engine.evaluate(0.85, amount=1200.0)
    assert res_high["risk_level"] == "HIGH"
    assert res_high["decision"] == "BLOCK"


def test_decision_engine_custom_thresholds():
    engine = FraudDecisionEngine(low_risk_threshold=0.15, high_risk_threshold=0.60)
    assert engine.evaluate(0.10)["decision"] == "ALLOW"
    assert engine.evaluate(0.25)["decision"] == "REVIEW"
    assert engine.evaluate(0.65)["decision"] == "BLOCK"


def test_decision_engine_invalid_thresholds():
    with pytest.raises(ValueError):
        FraudDecisionEngine(low_risk_threshold=0.80, high_risk_threshold=0.20)

    with pytest.raises(ValueError):
        FraudDecisionEngine(low_risk_threshold=-0.1, high_risk_threshold=0.7)


def test_decision_engine_invalid_probabilities():
    engine = FraudDecisionEngine()
    with pytest.raises(ValueError):
        engine.evaluate(-0.01)

    with pytest.raises(ValueError):
        engine.evaluate(1.05)

    with pytest.raises(ValueError):
        engine.evaluate(float("nan"))
