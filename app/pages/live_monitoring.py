"""
Page 2: Live Transaction Monitoring
Provides an interactive transaction stream simulator with start/pause/step controls,
real-time scoring, and risk categorization feed.
"""

import time
import streamlit as st
import pandas as pd
from app.config import RISK_COLORS, DECISION_COLORS


def render_live_monitoring(store, decision_engine, simulator):
    st.markdown("## 📡 Real-Time Transaction Monitoring")
    st.markdown(
        "Simulate live transaction traffic by sequentially streaming out-of-time test transactions "
        "through the production scoring pipeline and decision engine."
    )

    # Simulator Status Bar
    status_col1, status_col2, status_col3 = st.columns([1, 1, 1])
    
    status_label = "🟢 Streaming Active" if simulator.is_running else ("🟡 Paused" if simulator.is_paused else "⚪ Idle / Stopped")
    status_col1.markdown(f"**Stream Status:** `{status_label}`")
    status_col2.markdown(f"**Progress:** `{simulator.current_index:,} / {simulator.total_stream_length:,}` transactions")
    
    stats = store.get_summary_statistics()
    suspicion_rate = stats["fraud_suspicion_rate"]
    status_col3.markdown(f"**Suspicion Rate:** `{suspicion_rate:.2f}%`")

    st.markdown("---")

    # Control Panel
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6 = st.columns(6)

    with ctrl_col1:
        if st.button("▶️ Start / Run Stream", use_container_width=True, disabled=simulator.is_running):
            simulator.start()
            st.session_state["streaming_active"] = True
            st.rerun()

    with ctrl_col2:
        if st.button("⏸️ Pause", use_container_width=True, disabled=not simulator.is_running):
            simulator.pause()
            st.session_state["streaming_active"] = False
            st.rerun()

    with ctrl_col3:
        if st.button("⏹️ Stop", use_container_width=True, disabled=not (simulator.is_running or simulator.is_paused)):
            simulator.stop()
            st.session_state["streaming_active"] = False
            st.rerun()

    with ctrl_col4:
        if st.button("⏭️ Step (1 Txn)", use_container_width=True):
            res = simulator.step()
            st.rerun()

    with ctrl_col5:
        if st.button("⏩ Step Batch (20)", use_container_width=True):
            simulator.step_batch(20)
            st.rerun()

    with ctrl_col6:
        if st.button("🔄 Reset Stream", use_container_width=True):
            simulator.reset()
            st.session_state["streaming_active"] = False
            st.success("Transaction stream and memory buffer reset.")
            st.rerun()

    # Streaming Options Expander
    with st.expander("⚙️ Stream Simulation Settings", expanded=False):
        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            batch_burst = st.slider("Burst Transactions per Cycle", min_value=1, max_value=50, value=5, step=1)
        with opt_col2:
            delay_sec = st.slider("Cycle Delay (Seconds)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

    # Active stream execution cycle
    if st.session_state.get("streaming_active", False) and simulator.is_running:
        if not simulator.is_finished:
            # Process burst
            simulator.step_batch(batch_burst)
            time.sleep(delay_sec)
            st.rerun()
        else:
            simulator.stop()
            st.session_state["streaming_active"] = False
            st.info("Reached end of test transaction stream.")

    # Live Feed Section
    st.markdown("### 📋 Live Ingestion Feed (Recent 30)")
    recent = store.get_recent_transactions(limit=30)

    if recent:
        feed_data = []
        for r in recent:
            feed_data.append({
                "Txn ID": r["transaction_id"],
                "Elapsed (s)": f"{r['elapsed_time']:,.1f}",
                "Amount": f"${r['amount']:,.2f}",
                "Fraud Prob": f"{r['fraud_probability']:.2%}",
                "Risk Level": r["risk_level"],
                "Decision": r["decision"],
                "Status": r["review_status"],
            })
        feed_df = pd.DataFrame(feed_data)

        # Style with simple visual pills
        def color_risk(val):
            color = RISK_COLORS.get(val, "#000")
            return f"color: {color}; font-weight: bold;"

        st.dataframe(
            feed_df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Txn ID": st.column_config.TextColumn("Txn ID", width="small"),
                "Amount": st.column_config.TextColumn("Amount ($)", width="small"),
                "Fraud Prob": st.column_config.TextColumn("Fraud Score", width="small"),
                "Risk Level": st.column_config.TextColumn("Risk Level", width="small"),
                "Decision": st.column_config.TextColumn("Action Decision", width="small"),
            }
        )
    else:
        st.info("No incoming transactions yet. Click **▶️ Start Stream** or **⏭️ Step** above.")
