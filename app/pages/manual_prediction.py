"""
Page 4: Manual Fraud Prediction
Allows users and risk engineers to score individual custom or sample transactions,
inspect the feature vector, and observe decision engine reasoning.
"""

import streamlit as st
import pandas as pd
import numpy as np
from app.config import (
    FEATURE_COLUMNS,
    TIME_COLUMN,
    AMOUNT_COLUMN,
    PCA_FEATURES,
    RISK_COLORS,
    DECISION_COLORS,
)


def render_manual_prediction(predictor, decision_engine, store, simulator):
    st.markdown("## 🔍 Manual Transaction Risk Scoring")
    st.markdown(
        "Evaluate individual transaction feature vectors against the production ML pipeline and decision engine. "
        "You can manually adjust values or load historical sample transactions from the held-out test dataset."
    )

    # 1. Preset Sample Loaders
    st.markdown("### ⚡ Quick Presets (Held-Out Test Set)")
    preset_col1, preset_col2, preset_col3 = st.columns([1, 1, 1])

    if "manual_features" not in st.session_state:
        # Default zero-centered features
        init_data = {col: 0.0 for col in FEATURE_COLUMNS}
        init_data[AMOUNT_COLUMN] = 45.00
        init_data[TIME_COLUMN] = 7200.0
        st.session_state["manual_features"] = init_data

    with preset_col1:
        if st.button("🟢 Load Real Legitimate Transaction", use_container_width=True):
            if simulator._test_df is not None:
                legit_samples = simulator._test_df[simulator._test_df["Class"] == 0]
                if not legit_samples.empty:
                    chosen = legit_samples.sample(1, random_state=int(np.random.randint(1, 1000))).iloc[0]
                    st.session_state["manual_features"] = {c: float(chosen[c]) for c in FEATURE_COLUMNS}
                    st.session_state["preset_note"] = "Loaded historical legitimate sample (Class = 0)."
                    st.rerun()

    with preset_col2:
        if st.button("🔴 Load Real Fraudulent Transaction", use_container_width=True):
            if simulator._test_df is not None:
                fraud_samples = simulator._test_df[simulator._test_df["Class"] == 1]
                if not fraud_samples.empty:
                    chosen = fraud_samples.sample(1, random_state=int(np.random.randint(1, 1000))).iloc[0]
                    st.session_state["manual_features"] = {c: float(chosen[c]) for c in FEATURE_COLUMNS}
                    st.session_state["preset_note"] = "Loaded historical fraudulent sample (Class = 1)."
                    st.rerun()

    with preset_col3:
        if st.button("🔄 Reset to Default Baseline", use_container_width=True):
            init_data = {col: 0.0 for col in FEATURE_COLUMNS}
            init_data[AMOUNT_COLUMN] = 45.00
            init_data[TIME_COLUMN] = 7200.0
            st.session_state["manual_features"] = init_data
            st.session_state.pop("preset_note", None)
            st.rerun()

    if "preset_note" in st.session_state:
        st.info(st.session_state["preset_note"])

    # 2. Input Form
    st.markdown("### 📝 Transaction Features")
    features = st.session_state["manual_features"]

    with st.form("manual_prediction_form"):
        # Primary attributes
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            amount_val = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                max_value=100000.0,
                value=float(features.get(AMOUNT_COLUMN, 45.0)),
                step=1.0,
                format="%.2f",
            )
        with f_col2:
            time_val = st.number_input(
                "Elapsed Seconds from Inception (Time)",
                min_value=0.0,
                max_value=200000.0,
                value=float(features.get(TIME_COLUMN, 7200.0)),
                step=10.0,
            )

        # High Impact PCA features (e.g. top correlated features: V17, V14, V12, V10, V4, V11)
        st.markdown("##### High-Impact Diagnostic PCA Components")
        top_pca = ["V17", "V14", "V12", "V10", "V4", "V11"]
        top_cols = st.columns(3)
        top_vals = {}
        for idx, feat in enumerate(top_pca):
            col_target = top_cols[idx % 3]
            with col_target:
                top_vals[feat] = st.number_input(
                    f"{feat}",
                    value=float(features.get(feat, 0.0)),
                    step=0.1,
                    format="%.4f",
                    key=f"input_{feat}",
                )

        # Remaining PCA features in accordion
        with st.expander("Expand Other PCA Features (V1 to V28)", expanded=False):
            other_pca = [f for f in PCA_FEATURES if f not in top_pca]
            other_cols = st.columns(4)
            other_vals = {}
            for idx, feat in enumerate(other_pca):
                col_target = other_cols[idx % 4]
                with col_target:
                    other_vals[feat] = st.number_input(
                        f"{feat}",
                        value=float(features.get(feat, 0.0)),
                        step=0.1,
                        format="%.4f",
                        key=f"input_{feat}",
                    )

        submit_btn = st.form_submit_button("⚡ Evaluate Transaction Risk", use_container_width=True)

    if submit_btn:
        # Assemble feature dict
        payload = {TIME_COLUMN: time_val, AMOUNT_COLUMN: amount_val}
        payload.update(top_vals)
        payload.update(other_vals)

        # Run inference
        try:
            prob = predictor.predict_probability(payload)
            decision = decision_engine.evaluate(prob, amount=amount_val)

            st.markdown("---")
            st.markdown("### 🎯 Decision & Risk Assessment")

            res1, res2, res3 = st.columns(3)
            risk = decision["risk_level"]
            action = decision["decision"]

            res1.metric("Predicted Fraud Probability", f"{prob:.2%}")
            res2.metric("Assigned Risk Band", risk)
            res3.metric("Decision Engine Action", action)

            # Visual alert banner
            risk_color = RISK_COLORS.get(risk, "#333")
            st.markdown(
                f"""
                <div style="padding: 16px; border-radius: 8px; background-color: {risk_color}22; border-left: 6px solid {risk_color}; margin-top: 10px;">
                    <h4 style="margin: 0; color: {risk_color};">Outcome: {action} ({risk} RISK)</h4>
                    <p style="margin: 8px 0 0 0; font-size: 15px;">{decision['explanation']}</p>
                    <p style="margin: 4px 0 0 0; font-size: 13px; color: #555;"><strong>Action:</strong> {decision['action']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Option to record in store
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📥 Commit Evaluation to Live Dashboard Stream"):
                custom_id = f"MANUAL-{int(time_val)}"
                store.add_transaction(
                    transaction_id=custom_id,
                    amount=amount_val,
                    fraud_probability=prob,
                    risk_level=risk,
                    decision=action,
                    elapsed_time=time_val,
                    metadata={"source": "Manual Evaluation Form"}
                )
                st.success(f"Added transaction `{custom_id}` to memory store.")

        except Exception as e:
            st.error(f"Inference error: {e}")
