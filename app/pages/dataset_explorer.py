"""
Page 6: Dataset Explorer & Historical EDA
Provides interactive exploration of the historical ULB Credit Card Fraud dataset,
including class imbalance analysis, amount distributions, time trends, and correlation heatmaps.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from app.config import DATASET_PATH, TARGET_COLUMN, TIME_COLUMN, AMOUNT_COLUMN, PCA_FEATURES
from ml.preprocessing import load_dataset, get_dataset_summary


@st.cache_data
def load_dataset_cached():
    return load_dataset(DATASET_PATH)


def render_dataset_explorer():
    st.markdown("## 🗃️ Historical Dataset Explorer & EDA")
    st.markdown(
        "Interactive analysis of the ULB Machine Learning Group benchmark dataset (284,807 credit card transactions "
        "recorded over two days in September 2013 by European cardholders)."
    )

    try:
        df = load_dataset_cached()
    except Exception as e:
        st.error(f"Error loading historical dataset: {e}")
        return

    summary = get_dataset_summary(df)

    # 1. High-Level Dataset KPI Row
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Total Transactions", f"{summary['total_transactions']:,}")
    d2.metric("Legitimate (Class 0)", f"{summary['legitimate_transactions']:,}")
    d3.metric("Fraud (Class 1)", f"{summary['fraudulent_transactions']:,}")
    d4.metric("Fraud Percentage", f"{summary['fraud_percentage']:.4f}%")
    d5.metric("Missing Values", f"{summary['missing_values']:,}")

    st.markdown("---")

    # 2. Imbalance & Amount Distribution Charts
    c_imb, c_amt = st.columns([1, 1])

    with c_imb:
        st.markdown("### Class Imbalance (Log Scale)")
        imb_data = pd.DataFrame({
            "Transaction Type": ["Legitimate (0)", "Fraudulent (1)"],
            "Count": [summary["legitimate_transactions"], summary["fraudulent_transactions"]],
        })
        fig_imb = px.bar(
            imb_data,
            x="Transaction Type",
            y="Count",
            color="Transaction Type",
            color_discrete_sequence=["#10B981", "#EF4444"],
            log_y=True,
            text="Count",
        )
        fig_imb.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_imb.update_layout(height=320, margin=dict(t=30, b=20, l=20, r=20), showlegend=False)
        st.plotly_chart(fig_imb, use_container_width=True)

    with c_amt:
        st.markdown("### Transaction Amount Distribution (Log X)")
        sample_df = df.sample(min(15000, len(df)), random_state=42).copy()
        sample_df["Type"] = sample_df["Class"].map({0: "Legitimate", 1: "Fraud"})
        fig_amt = px.histogram(
            sample_df,
            x="Amount",
            color="Type",
            color_discrete_map={"Legitimate": "#10B981", "Fraud": "#EF4444"},
            log_x=True,
            barmode="overlay",
            nbins=30,
        )
        fig_amt.update_layout(height=320, margin=dict(t=30, b=20, l=20, r=20))
        st.plotly_chart(fig_amt, use_container_width=True)

    # 3. Top Correlations with Fraud
    st.markdown("### 🧬 Feature Correlations with Fraud Label (Class)")
    correlations = df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values()
    top_negative = correlations.head(6)
    top_positive = correlations.tail(6)
    top_corr = pd.concat([top_negative, top_positive]).reset_index()
    top_corr.columns = ["Feature", "Correlation"]

    fig_corr = px.bar(
        top_corr,
        x="Correlation",
        y="Feature",
        orientation="h",
        color="Correlation",
        color_continuous_scale="RdBu_r",
    )
    fig_corr.update_layout(height=340, margin=dict(t=30, b=20, l=20, r=20))
    st.plotly_chart(fig_corr, use_container_width=True)

    # 4. Filterable Dataset Table Sample
    st.markdown("### 🔍 Historical Transaction Data Browser")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        class_choice = st.radio("Filter by Class Label", ["All", "Legitimate Only (0)", "Fraud Only (1)"], horizontal=True)
    with col_f2:
        max_amt_filter = st.slider("Max Amount Filter ($)", min_value=10.0, max_value=5000.0, value=2500.0, step=50.0)

    filtered = df[df["Amount"] <= max_amt_filter]
    if class_choice == "Legitimate Only (0)":
        filtered = filtered[filtered["Class"] == 0]
    elif class_choice == "Fraud Only (1)":
        filtered = filtered[filtered["Class"] == 1]

    st.markdown(f"Displaying **{min(100, len(filtered))}** sample rows (out of {len(filtered):,} matching):")
    st.dataframe(filtered.head(100), height=300, use_container_width=True)
