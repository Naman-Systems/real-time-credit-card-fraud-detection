"""
Model Evaluation and Metrics Module for Credit Card Fraud Detection.
Calculates classification metrics, confusion matrices, ROC/PR curves,
and performs threshold optimization.
"""

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics at a given threshold.
    
    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted fraud probabilities [0, 1].
        threshold: Decision threshold for positive class assignment.
        
    Returns:
        Dictionary of performance metrics.
    """
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Derived rates
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    # Curve AUCs (threshold-independent)
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "total_samples": int(len(y_true)),
        "total_actual_frauds": int(np.sum(y_true)),
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold_candidates: List[float] = None,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluate multiple decision thresholds on validation data to select the optimal operating point.
    
    Args:
        y_true: Validation ground truth.
        y_prob: Predicted validation probabilities.
        threshold_candidates: List of thresholds to test (default 0.10 to 0.90 in steps of 0.05).
        
    Returns:
        Tuple of (best_threshold, list of evaluation dictionaries per threshold).
    """
    if threshold_candidates is None:
        threshold_candidates = [
            0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90
        ]

    evaluations: List[Dict[str, Any]] = []
    best_threshold = 0.5
    best_f1 = -1.0

    for thresh in threshold_candidates:
        metrics = compute_classification_metrics(y_true, y_prob, threshold=thresh)
        evaluations.append(metrics)
        # Select threshold maximizing F1-score (harmonic mean of precision and recall)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_threshold = thresh

    return best_threshold, evaluations


def get_curve_points(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_points: int = 500,
) -> Dict[str, Any]:
    """
    Generate subsampled PR and ROC curve coordinate data for plotting without memory bloat.
    
    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted fraud probabilities.
        max_points: Max points to preserve for smooth plotting.
        
    Returns:
        Dictionary with roc and pr curve data.
    """
    # Precision-Recall
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
    # ROC
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)

    def subsample(arr1, arr2, limit):
        if len(arr1) <= limit:
            return arr1.tolist(), arr2.tolist()
        step = len(arr1) // limit
        return arr1[::step].tolist(), arr2[::step].tolist()

    rec_sub, prec_sub = subsample(recalls, precisions, max_points)
    fpr_sub, tpr_sub = subsample(fpr, tpr, max_points)

    return {
        "pr_curve": {
            "recalls": rec_sub,
            "precisions": prec_sub,
            "pr_auc": float(average_precision_score(y_true, y_prob)),
        },
        "roc_curve": {
            "fpr": fpr_sub,
            "tpr": tpr_sub,
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
        },
    }
