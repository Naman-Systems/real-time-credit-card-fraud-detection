"""
Page 1: Dashboard Overview
Executive risk monitoring view with live KPI cards, interactive Plotly visualizations,
and recent transaction feed.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from app.config import RISK_COLORS, DECISION_COLORS


def render_dashboard(store, decision_engine, simulator):
    st.markdown("## 📊 Executive Risk Monitoring Dashboard")
    st.markdown(
        "Real-time operational overview of streamed credit card transactions, risk categorizations, "
        "and automated fraud prevention decisions."
    )

    stats = store.get_summary_statistics()
    recent_txns = store.get_recent_transactions(limit=20)

    # 1. KPI Cards Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Total Processed", f"{stats['total_processed']:,}")
    kpi2.metric("Total Volume ($)", f"${stats['total_amount']:,.2f}")
    kpi3.metric("Allowed (Legit)", f"{stats['allow_count']:,}", delta=f"{stats['allow_count']/(stats['total_processed'] or 1):.1%}")
    kpi4.metric("Flagged / Review", f"{stats['review_count']:,}", delta_color="inverse")
    kpi5.metric("Blocked (Fraud)", f"{stats['block_count']:,}", delta_color="inverse")

    st.markdown("---")

    # 2. Charts Row
    if stats["total_processed"] > 0:
        tx_df = store.to_dataframe()

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("### Decision Distribution")
            decision_counts = tx_df["decision"].value_counts().reset_index()
            decision_counts.columns = ["Decision", "Count"]
            
            color_map = {
                "ALLOW": DECISION_COLORS["ALLOW"],
                "REVIEW": DECISION_COLORS["REVIEW"],
                "BLOCK": DECISION_COLORS["BLOCK"],
            }

            fig_decision = px.pie(
                decision_counts,
                names="Decision",
                values="Count",
                color="Decision",
                color_discrete_map=color_map,
                hole=0.45,
            )
            fig_decision.update_traces(textinfo="percent+label", hoverinfo="label+value+percent")
            fig_decision.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
            st.plotly_chart(fig_decision, use_container_width=True)

        with col_right:
            st.markdown("### Fraud Probability Distribution")
            fig_hist = px.histogram(
                tx_df,
                x="fraud_probability",
                nbins=25,
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                labels={"fraud_probability": "Predicted Fraud Probability", "risk_level": "Risk Category"},
            )
            fig_hist.add_vline(x=decision_engine.low_threshold, line_dash="dash", line_color=RISK_COLORS["MEDIUM"], annotation_text="Review Thresh")
            fig_hist.add_vline(x=decision_engine.high_threshold, line_dash="dash", line_color=RISK_COLORS["HIGH"], annotation_text="Block Thresh")
            fig_hist.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, bargap=0.05)
            st.plotly_chart(fig_hist, use_container_width=True)

        # Activity Over Time & Amount Distribution
        col_act, col_amt = st.columns([1, 1])

        with col_act:
            st.markdown("### Cumulative Transactions Processed")
            tx_df_rev = tx_df.sort_index(ascending=True).reset_index()
            tx_df_rev["cumulative_count"] = tx_df_rev.index + 1
            fig_time = px.line(
                tx_df_rev,
                x="cumulative_count",
                y="fraud_probability",
                color="decision",
                color_discrete_map=DECISION_COLORS,
                labels={"cumulative_count": "Processed Sequence", "fraud_probability": "Fraud Probability"},
                title="Transaction Stream Probability Sequence",
            )
            fig_time.update_layout(height=300, margin=dict(t=30, b=20, l=20, r=20))
            st.plotly_chart(fig_time, use_container_width=True)

        with col_amt:
            st.markdown("### Transaction Amount by Risk Category")
            fig_box = px.strip(
                tx_df,
                x="risk_level",
                y="amount",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                labels={"risk_level": "Risk Category", "amount": "Amount ($)"},
            )
            fig_box.update_layout(height=300, margin=dict(t=30, b=20, l=20, r=20))
            st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.info(
            "💡 No transactions currently in memory. Navigate to **Live Monitoring** to start the replay "
            "stream, or visit **Manual Prediction** to score individual transactions."
        )

    # 3. Recent Transactions Feed
    st.markdown("### 🕒 Recent Processed Transactions")
    if recent_txns:
        feed_df = pd.DataFrame(recent_txns)[
            ["transaction_id", "timestamp", "amount", "fraud_probability", "risk_level", "decision", "review_status"]
        ]
        feed_df["amount"] = feed_df["amount"].apply(lambda v: f"${v:,.2f}")
        feed_df["fraud_probability"] = feed_df["fraud_probability"].apply(lambda v: f"{v:.2%}")
        
        st.dataframe(
            feed_df,
            column_config={
                "transaction_id": "Transaction ID",
                "timestamp": "Timestamp",
                "amount": "Amount",
                "fraud_probability": "Fraud Score",
                "risk_level": "Risk Level",
                "decision": "Decision",
                "review_status": "Status",
            },
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.caption("Awaiting incoming transactions...")
