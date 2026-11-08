"""
Model Training and Experimentation Pipeline for Credit Card Fraud Detection.
Trains baseline, tree, ensemble, and SMOTE models, compares their validation performance,
optimizes decision thresholds, and exports production artifacts.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from app.config import (
    DATASET_PATH,
    MODEL_PATH,
    METADATA_PATH,
    EVALUATION_METRICS_PATH,
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    DEFAULT_CLASSIFICATION_THRESHOLD,
)
from ml.preprocessing import (
    load_dataset,
    chronological_split,
    build_feature_preprocessor,
    validate_inference_features,
)
from ml.evaluate import (
    compute_classification_metrics,
    optimize_threshold,
    get_curve_points,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ml.train")


def train_and_evaluate_all_models(
    dataset_path: Path = DATASET_PATH
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, Any]]:
    """
    Execute end-to-end model training, validation comparison, and artifact generation.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load historical dataset
    df = load_dataset(dataset_path)

    # 2. Chronological split (70% train, 15% val, 15% test)
    train_df, val_df, test_df = chronological_split(df, 0.70, 0.15, 0.15)

    X_train = validate_inference_features(train_df)
    y_train = train_df[TARGET_COLUMN].values

    X_val = validate_inference_features(val_df)
    y_val = val_df[TARGET_COLUMN].values

    X_test = validate_inference_features(test_df)
    y_test = test_df[TARGET_COLUMN].values

    # Save test partition for out-of-time simulation replay
    test_partition_path = ARTIFACTS_DIR / "held_out_test.csv"
    test_df.to_csv(test_partition_path, index=False)
    logger.info("Saved held-out test split (%d rows) to %s", len(test_df), test_partition_path)

    # 3. Define candidate pipelines
    # Candidate A: Logistic Regression Baseline (Class Weighted)
    lr_pipe = Pipeline([
        ("preprocessor", build_feature_preprocessor()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            solver="lbfgs"
        ))
    ])

    # Candidate B: Decision Tree (Interpretable Tree)
    dt_pipe = Pipeline([
        ("preprocessor", build_feature_preprocessor()),
        ("classifier", DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Candidate C: Random Forest Ensemble (Balanced Subsample)
    rf_pipe = Pipeline([
        ("preprocessor", build_feature_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Candidate D: SMOTE + Random Forest
    smote_rf_pipe = ImbPipeline([
        ("preprocessor", build_feature_preprocessor()),
        ("smote", SMOTE(random_state=42, sampling_strategy=0.1)),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ))
    ])

    models = {
        "Logistic Regression": lr_pipe,
        "Decision Tree": dt_pipe,
        "Random Forest": rf_pipe,
        "Random Forest + SMOTE": smote_rf_pipe,
    }

    results: Dict[str, Any] = {}
    fitted_pipelines: Dict[str, Any] = {}

    logger.info("Starting candidate model training and validation...")
    for name, pipeline in models.items():
        logger.info("Training candidate: %s ...", name)
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predict probability on validation set
        val_probs = pipeline.predict_proba(X_val)[:, 1]
        val_metrics_default = compute_classification_metrics(y_val, val_probs, threshold=0.5)
        
        # Optimize threshold
        best_thresh, thresh_sweep = optimize_threshold(y_val, val_probs)
        val_metrics_opt = compute_classification_metrics(y_val, val_probs, threshold=best_thresh)

        # PR and ROC curves
        curves = get_curve_points(y_val, val_probs)

        results[name] = {
            "default_metrics": val_metrics_default,
            "optimal_threshold": best_thresh,
            "optimized_metrics": val_metrics_opt,
            "threshold_sweep": thresh_sweep,
            "curves": curves,
        }
        logger.info(
            "Candidate [%s] - Val PR-AUC: %.4f | F1 (0.50): %.4f | F1 (Opt @ %.2f): %.4f | Recall: %.4f | Precision: %.4f",
            name,
            val_metrics_default["pr_auc"],
            val_metrics_default["f1_score"],
            best_thresh,
            val_metrics_opt["f1_score"],
            val_metrics_opt["recall"],
            val_metrics_opt["precision"],
        )

    # 4. Model Selection based on Validation PR-AUC and Optimized F1
    supervised_candidates = ["Logistic Regression", "Decision Tree", "Random Forest", "Random Forest + SMOTE"]
    best_candidate_name = max(
        supervised_candidates,
        key=lambda m: (results[m]["default_metrics"]["pr_auc"], results[m]["optimized_metrics"]["f1_score"])
    )

    best_pipeline = fitted_pipelines[best_candidate_name]
    best_threshold = results[best_candidate_name]["optimal_threshold"]
    best_val_metrics = results[best_candidate_name]["optimized_metrics"]

    logger.info(">>> SELECTED BEST MODEL: %s (Optimal Threshold: %.2f) <<<", best_candidate_name, best_threshold)

    # 5. Final Out-Of-Time Evaluation on Held-Out Test Set
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    test_metrics_default = compute_classification_metrics(y_test, test_probs, threshold=0.5)
    test_metrics_opt = compute_classification_metrics(y_test, test_probs, threshold=best_threshold)
    test_curves = get_curve_points(y_test, test_probs)

    logger.info(
        "Final Held-Out Test Performance for %s:\n"
        "  - Optimal Threshold: %.2f\n"
        "  - Test PR-AUC: %.4f\n"
        "  - Test ROC-AUC: %.4f\n"
        "  - Test Recall: %.4f (Frauds caught: %d/%d)\n"
        "  - Test Precision: %.4f\n"
        "  - Test F1-Score: %.4f\n"
        "  - False Positives: %d\n"
        "  - False Negatives: %d",
        best_candidate_name,
        best_threshold,
        test_metrics_opt["pr_auc"],
        test_metrics_opt["roc_auc"],
        test_metrics_opt["recall"],
        test_metrics_opt["true_positives"],
        test_metrics_opt["total_actual_frauds"],
        test_metrics_opt["precision"],
        test_metrics_opt["f1_score"],
        test_metrics_opt["false_positives"],
        test_metrics_opt["false_negatives"],
    )

    # Extract feature importances if Random Forest or Decision Tree
    feature_importances = {}
    try:
        classifier = best_pipeline.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            # Map back to feature columns
            # preprocessor output order: [Amount, Time, V1...V28]
            feature_order = [X_train.columns[30 - 1], X_train.columns[0]] + list(X_train.columns[1:29])
            # To be exact, preprocessor orders: Amount (idx 0), Time (idx 1), V1-V28 (idx 2-29)
            trans_feature_names = ["Amount", "Time"] + [f"V{i}" for i in range(1, 29)]
            for fname, imp in zip(trans_feature_names, classifier.feature_importances_):
                feature_importances[fname] = float(imp)
            # Sort descending
            feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    except Exception as e:
        logger.warning("Could not extract feature importances: %s", e)

    # 6. Save Model Artifact
    joblib.dump(best_pipeline, MODEL_PATH)
    logger.info("Saved fitted production pipeline to %s", MODEL_PATH)

    # 7. Metadata Artifact
    metadata = {
        "model_name": best_candidate_name,
        "model_version": "1.0.0",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "feature_count": len(FEATURE_COLUMNS),
        "feature_names": FEATURE_COLUMNS,
        "selected_threshold": best_threshold,
        "feature_importances": feature_importances,
        "validation_metrics": best_val_metrics,
        "test_metrics": test_metrics_opt,
        "test_metrics_default_threshold": test_metrics_default,
        "hyperparameters": {
            k: str(v) for k, v in best_pipeline.named_steps["classifier"].get_params().items()
        },
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata artifact to %s", METADATA_PATH)

    # 8. Evaluation Metrics Artifact (for UI model comparison page)
    # Simplify structure for JSON serialization
    evaluation_export = {
        "comparison": {
            model_name: {
                "default_metrics": data["default_metrics"],
                "optimal_threshold": data["optimal_threshold"],
                "optimized_metrics": data["optimized_metrics"],
                "curves": data["curves"],
                "threshold_sweep": data.get("threshold_sweep", []),
            }
            for model_name, data in results.items()
        },
        "selected_model": best_candidate_name,
        "optimal_threshold": best_threshold,
        "final_test_metrics": test_metrics_opt,
        "test_curves": test_curves,
    }

    with open(EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_export, f, indent=2)
    logger.info("Saved evaluation metrics artifact to %s", EVALUATION_METRICS_PATH)

    return best_pipeline, metadata, evaluation_export


if __name__ == "__main__":
    train_and_evaluate_all_models()
