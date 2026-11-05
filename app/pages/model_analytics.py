"""
Page 5: Model Analytics & Offline Performance Evaluation
Displays comparative benchmark metrics, confusion matrix, ROC / Precision-Recall curves,
threshold optimization curves, and tree feature importance.
"""

import json
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app.config import METADATA_PATH, EVALUATION_METRICS_PATH


@st.cache_data
def load_offline_analytics():
    metadata = {}
    metrics_data = {}
    if METADATA_PATH.is_file():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    if EVALUATION_METRICS_PATH.is_file():
        with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
    return metadata, metrics_data


def render_model_analytics():
    st.markdown("## 📈 Offline Model Performance & Evaluation Analytics")
    st.markdown(
        "Detailed performance evaluation across baseline, tree, ensemble, and anomaly detection algorithms, "
        "trained strictly on chronological data (70% train, 15% validation, 15% out-of-time test)."
    )

    metadata, metrics_data = load_offline_analytics()

    if not metadata or not metrics_data:
        st.warning("Model evaluation artifacts not found. Please execute `python ml/train.py`.")
        return

    # Selected Model Banner
    selected_name = metadata.get("model_name", "Random Forest")
    threshold = metadata.get("selected_threshold", 0.40)
    test_metrics = metadata.get("test_metrics", {})
    val_metrics = metadata.get("validation_metrics", {})

    st.success(
        f"🏆 **Production Selected Model:** `{selected_name}` | "
        f"**Optimal Threshold:** `{threshold:.2f}` | "
        f"**Out-of-Time Test PR-AUC:** `{test_metrics.get('pr_auc', 0.0):.4f}` | "
        f"**Test F1-Score:** `{test_metrics.get('f1_score', 0.0):.4f}`"
    )

    # 1. Held-Out Test Metric Cards
    st.markdown("### 🎯 Held-Out Test Performance (42,722 Unseen Future Transactions)")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PR-AUC (Primary)", f"{test_metrics.get('pr_auc', 0):.4f}")
    c2.metric("Recall (Fraud Catch)", f"{test_metrics.get('recall', 0):.2%}")
    c3.metric("Precision", f"{test_metrics.get('precision', 0):.2%}")
    c4.metric("F1-Score", f"{test_metrics.get('f1_score', 0):.4f}")
    c5.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.4f}")

    st.markdown("---")

    # 2. Model Comparison Table
    st.markdown("### ⚖️ Algorithm Benchmark Comparison (Validation Set)")
    comparison = metrics_data.get("comparison", {})
    comp_rows = []
    for model_name, info in comparison.items():
        opt = info.get("optimized_metrics", {})
        default = info.get("default_metrics", {})
        comp_rows.append({
            "Algorithm": model_name,
            "Val PR-AUC": default.get("pr_auc", 0.0),
            "Val ROC-AUC": default.get("roc_auc", 0.0),
            "Optimal Thresh": info.get("optimal_threshold", 0.5),
            "Recall @ Opt": opt.get("recall", 0.0),
            "Precision @ Opt": opt.get("precision", 0.0),
            "F1 @ Opt": opt.get("f1_score", 0.0),
            "False Positives": opt.get("false_positives", 0),
            "False Negatives": opt.get("false_negatives", 0),
        })

    comp_df = pd.DataFrame(comp_rows).sort_values(by="Val PR-AUC", ascending=False)
    st.dataframe(
        comp_df.style.format({
            "Val PR-AUC": "{:.4f}",
            "Val ROC-AUC": "{:.4f}",
            "Optimal Thresh": "{:.2f}",
            "Recall @ Opt": "{:.2%}",
            "Precision @ Opt": "{:.2%}",
            "F1 @ Opt": "{:.4f}",
        }).highlight_max(subset=["Val PR-AUC", "F1 @ Opt"], color="#10B98133"),
        use_container_width=True,
        hide_index=True,
    )

    # 3. Confusion Matrix and Feature Importances
    col_cm, col_imp = st.columns([1, 1])

    with col_cm:
        st.markdown(f"### Confusion Matrix (Test Set @ {threshold:.2f})")
        tp = test_metrics.get("true_positives", 0)
        fp = test_metrics.get("false_positives", 0)
        tn = test_metrics.get("true_negatives", 0)
        fn = test_metrics.get("false_negatives", 0)

        cm_matrix = [[tn, fp], [fn, tp]]
        fig_cm = px.imshow(
            cm_matrix,
            labels=dict(x="Predicted Class", y="Actual Class", color="Transaction Count"),
            x=["Legitimate (0)", "Fraud (1)"],
            y=["Legitimate (0)", "Fraud (1)"],
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig_cm.update_layout(height=340, margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_imp:
        st.markdown("### Top Feature Importances (Random Forest)")
        importances = metadata.get("feature_importances", {})
        if importances:
            top_feats = list(importances.items())[:10]
            top_df = pd.DataFrame(top_feats, columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=True)
            fig_imp = px.bar(
                top_df,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Viridis",
            )
            fig_imp.update_layout(height=340, margin=dict(t=30, b=20, l=20, r=20))
            st.plotly_chart(fig_imp, use_container_width=True)

    # 4. PR Curve and ROC Curve
    col_pr, col_roc = st.columns([1, 1])
    test_curves = metrics_data.get("test_curves", {})

    with col_pr:
        st.markdown("### Precision-Recall Curve")
        pr_data = test_curves.get("pr_curve", {})
        if pr_data:
            recalls = pr_data.get("recalls", [])
            precisions = pr_data.get("precisions", [])
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(
                x=recalls,
                y=precisions,
                mode="lines",
                name=f"PR Curve (AUC = {pr_data.get('pr_auc', 0):.4f})",
                line=dict(color="#10B981", width=2.5),
            ))
            fig_pr.update_layout(
                xaxis_title="Recall (Fraud Detection Rate)",
                yaxis_title="Precision (Positive Predictive Value)",
                height=320,
                margin=dict(t=30, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_pr, use_container_width=True)

    with col_roc:
        st.markdown("### ROC Curve")
        roc_data = test_curves.get("roc_curve", {})
        if roc_data:
            fpr = roc_data.get("fpr", [])
            tpr = roc_data.get("tpr", [])
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"ROC Curve (AUC = {roc_data.get('roc_auc', 0):.4f})",
                line=dict(color="#3B82F6", width=2.5),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random Guess", line=dict(dash="dash", color="#9CA3AF")
            ))
            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate (Recall)",
                height=320,
                margin=dict(t=30, b=20, l=20, r=20),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

    # 5. Threshold Optimization Sweep Analysis
    st.markdown("### 🎚️ Threshold Optimization Trade-off Analysis")
    rf_info = comparison.get(selected_name, {})
    sweep = rf_info.get("threshold_sweep", [])
    if sweep:
        sw_df = pd.DataFrame(sweep)
        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(x=sw_df["threshold"], y=sw_df["precision"], mode="lines+markers", name="Precision", line=dict(color="#3B82F6")))
        fig_sweep.add_trace(go.Scatter(x=sw_df["threshold"], y=sw_df["recall"], mode="lines+markers", name="Recall", line=dict(color="#EF4444")))
        fig_sweep.add_trace(go.Scatter(x=sw_df["threshold"], y=sw_df["f1_score"], mode="lines+markers", name="F1-Score", line=dict(color="#10B981", width=3)))
        fig_sweep.add_vline(x=threshold, line_dash="dash", line_color="#8B5CF6", annotation_text=f"Selected Thresh ({threshold:.2f})")
        fig_sweep.update_layout(
            xaxis_title="Classification Threshold",
            yaxis_title="Metric Value",
            height=340,
            margin=dict(t=30, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_sweep, use_container_width=True)

    # Educational Technical Callout
    with st.expander("📘 Technical Architecture & Evaluation Notes", expanded=False):
        st.markdown(
            """
            - **Why Accuracy is Misleading:** With 99.83% legitimate transactions, a trivial dummy baseline predicting 
              every transaction as legitimate achieves 99.83% accuracy while catching 0% of fraud. Therefore, 
              **Precision-Recall AUC (PR-AUC)** and **Recall / F1-Score** are our primary benchmark criteria.
            - **Chronological Split Leakage Prevention:** Standard k-fold cross-validation or random train-test splits 
              allow future transactions to bleed into training representations. Here, training (70%), validation (15%), 
              and testing (15%) are partitioned strictly in forward chronological sequence.
            - **Decision Threshold vs Risk Bands:** The ML threshold (0.40) determines optimal binary classification, 
              while the operational business decision engine uses multi-tier risk bands (ALLOW < 0.20, REVIEW 0.20-0.70, BLOCK >= 0.70) 
              to balance fraud deterrence against customer friction.
            """
        )


if __name__ == "__main__":
    from pathlib import Path
    import sys
    ROOT = Path(__file__).resolve().parent.parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app.streamlit_app import LOGO_PATH
    if LOGO_PATH.is_file():
        st.logo(str(LOGO_PATH), size="large")
    render_model_analytics()
