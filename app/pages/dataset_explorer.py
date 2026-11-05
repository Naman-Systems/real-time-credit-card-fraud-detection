"""
Page 6: Dataset Explorer & Historical EDA
Provides interactive exploration of the ULB Credit Card Fraud dataset.
On Streamlit Cloud (no raw dataset available), falls back to the bundled
compressed EDA sample (ml/artifacts/eda_sample.csv.gz) which contains
all 52 fraud transactions plus 10,000 stratified legitimate records.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
from app.config import (
    DATASET_PATH,
    EDA_SAMPLE_GZ_PATH,
    TARGET_COLUMN,
    TIME_COLUMN,
    AMOUNT_COLUMN,
    PCA_FEATURES,
)
from ml.preprocessing import get_dataset_summary


@st.cache_data
def load_dataset_cached():
    """
    Load the transaction dataset with a graceful three-tier fallback:
    1. Full raw creditcard.csv (local dev or mounted volume).
    2. Bundled compressed EDA sample ml/artifacts/eda_sample.csv.gz (Streamlit Cloud).
    3. Synthetic placeholder (pure cloud fallback with no real data).
    """
    # Tier 1: full raw dataset
    if DATASET_PATH.is_file():
        import pandas as pd
        df = pd.read_csv(DATASET_PATH)
        return df, False  # (dataframe, is_sample)

    # Tier 2: bundled compressed EDA sample
    if EDA_SAMPLE_GZ_PATH.is_file():
        import pandas as pd
        df = pd.read_csv(EDA_SAMPLE_GZ_PATH)
        return df, True  # is_sample=True — only 10k+52 rows

    # Tier 3: fully synthetic fallback
    import numpy as np
    import pandas as pd
    np.random.seed(42)
    n = 5000
    rows = {f: np.random.normal(0, 1, n) for f in PCA_FEATURES}
    rows[AMOUNT_COLUMN] = np.abs(np.random.exponential(scale=65.0, size=n))
    rows[TIME_COLUMN] = np.linspace(0, 172800, n)
    rows[TARGET_COLUMN] = (np.random.rand(n) < 0.0017).astype(int)
    df = pd.DataFrame(rows)
    return df, True


def render_dataset_explorer():
    st.markdown("## 🗃️ Historical Dataset Explorer & EDA")
    st.markdown(
        "Interactive analysis of the ULB Machine Learning Group benchmark dataset — "
        "284,807 credit card transactions recorded over two days in September 2013 "
        "by European cardholders."
    )

    try:
        df, is_sample = load_dataset_cached()
    except Exception as e:
        st.error(
            f"⚠️ Could not load any dataset source: `{e}`\n\n"
            "On Streamlit Cloud the bundled EDA sample (`ml/artifacts/eda_sample.csv.gz`) "
            "should be present. If you are running locally, place `creditcard.csv` in the "
            "`dataset/` directory."
        )
        return

    # Cloud-mode banner
    if is_sample:
        st.info(
            "📦 **Cloud Mode — Showing Stratified EDA Sample**\n\n"
            "The full 150 MB raw dataset (`creditcard.csv`) is not bundled with this "
            "deployment. All charts and correlations below are computed from the "
            "bundled EDA sample: **all 52 fraud transactions + 10,000 randomly sampled "
            "legitimate transactions** (out of the original 284,315).\n\n"
            "To run the full explorer locally, download the dataset from Kaggle and "
            "place it in the `dataset/` directory.",
            icon="ℹ️",
        )

    summary = get_dataset_summary(df)

    # 1. High-Level Dataset KPI Row
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Transactions (Sample)", f"{summary['total_transactions']:,}")
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
        sample_df = df.sample(min(10000, len(df)), random_state=42).copy()
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
    numeric_df = df.select_dtypes(include="number")
    if TARGET_COLUMN in numeric_df.columns:
        correlations = numeric_df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).sort_values()
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
        class_choice = st.radio(
            "Filter by Class Label",
            ["All", "Legitimate Only (0)", "Fraud Only (1)"],
            horizontal=True,
        )
    with col_f2:
        max_amt_filter = st.slider(
            "Max Amount Filter ($)",
            min_value=10.0,
            max_value=5000.0,
            value=2500.0,
            step=50.0,
        )

    filtered = df[df["Amount"] <= max_amt_filter]
    if class_choice == "Legitimate Only (0)":
        filtered = filtered[filtered["Class"] == 0]
    elif class_choice == "Fraud Only (1)":
        filtered = filtered[filtered["Class"] == 1]

    st.markdown(f"Displaying **{min(100, len(filtered))}** sample rows (out of {len(filtered):,} matching):")
    st.dataframe(filtered.head(100), height=300, use_container_width=True)


if __name__ == "__main__":
    from pathlib import Path
    import sys
    ROOT = Path(__file__).resolve().parent.parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app.streamlit_app import LOGO_PATH
    if LOGO_PATH.is_file():
        st.logo(str(LOGO_PATH), size="large")
    render_dataset_explorer()
