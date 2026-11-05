"""
Page 2: Live Transaction Monitoring
Provides an interactive transaction stream simulator with start/pause/step controls,
real-time scoring, and an Excel/CSV batch spreadsheet ingestion pipeline with automated
feature validation, risk scoring, live streaming replay, and alert queuing.
"""

import io
import time
from typing import Optional
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.config import (
    AMOUNT_COLUMN,
    DECISION_COLORS,
    FEATURE_COLUMNS,
    PCA_FEATURES,
    RISK_COLORS,
    TARGET_COLUMN,
    TIME_COLUMN,
)
from ml.predictor import FraudPredictor, get_predictor


def generate_sample_template(simulator=None, file_format: str = "xlsx") -> bytes:
    """
    Generate a sample transaction spreadsheet preloaded with realistic test transactions.
    """
    sample_df = None
    if simulator is not None and getattr(simulator, "_test_df", None) is not None:
        test_df = simulator._test_df
        legit = test_df[test_df[TARGET_COLUMN] == 0].head(5)
        fraud = test_df[test_df[TARGET_COLUMN] == 1].head(5)
        sample_df = pd.concat([legit, fraud], ignore_index=True)
    
    if sample_df is None or sample_df.empty:
        rows = []
        for i in range(10):
            row = {col: 0.0 for col in FEATURE_COLUMNS}
            row[AMOUNT_COLUMN] = float((i + 1) * 35.50)
            row[TIME_COLUMN] = float(i * 120)
            if i >= 5:  # make a few simulated fraud-like outliers
                row["V14"] = -8.2
                row["V17"] = -6.5
                row["V12"] = -5.0
                row[TARGET_COLUMN] = 1
            else:
                row[TARGET_COLUMN] = 0
            rows.append(row)
        sample_df = pd.DataFrame(rows)

    if file_format == "xlsx":
        buffer = io.BytesIO()
        sample_df.to_excel(buffer, index=False, engine="openpyxl")
        return buffer.getvalue()
    else:
        return sample_df.to_csv(index=False).encode("utf-8")


def render_live_monitoring(store, decision_engine, simulator, predictor: Optional[FraudPredictor] = None):
    st.markdown("## 📡 Real-Time Transaction Monitoring")
    st.markdown(
        "Monitor transactional streams in real-time, replay out-of-time test partitions, "
        "or ingest external Excel / CSV spreadsheets for instant batch fraud risk scoring."
    )

    if predictor is None:
        predictor = getattr(simulator, "predictor", None) or get_predictor()

    if simulator is None:
        from app.simulator import TransactionSimulator
        try:
            simulator = TransactionSimulator(
                predictor=predictor,
                decision_engine=decision_engine,
                store=store,
            )
            st.session_state["simulator"] = simulator
        except Exception:
            simulator = None

    tab_stream, tab_excel = st.tabs(["📡 Live Replay Simulator", "📑 Excel / CSV Batch Predictor"])

    # =========================================================================
    # TAB 1: Live Replay Simulator
    # =========================================================================
    with tab_stream:
        # Simulator Status Bar
        status_col1, status_col2, status_col3 = st.columns([1, 1, 1])

        is_running = getattr(simulator, "is_running", False)
        is_paused = getattr(simulator, "is_paused", False)
        curr_idx = getattr(simulator, "current_index", 0)
        total_len = getattr(simulator, "total_stream_length", 0)

        status_label = (
            "🟢 Streaming Active"
            if is_running
            else ("🟡 Paused" if is_paused else "⚪ Idle / Stopped")
        )
        status_col1.markdown(f"**Stream Status:** `{status_label}`")
        status_col2.markdown(
            f"**Progress:** `{curr_idx:,} / {total_len:,}` transactions"
        )

        stats = store.get_summary_statistics() if store else {"fraud_suspicion_rate": 0.0}
        suspicion_rate = stats.get("fraud_suspicion_rate", 0.0)
        status_col3.markdown(f"**Suspicion Rate:** `{suspicion_rate:.2f}%`")

        st.markdown("---")

        # Control Panel
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6 = st.columns(6)

        with ctrl_col1:
            if st.button("▶️ Start / Run Stream", use_container_width=True, disabled=is_running or simulator is None):
                if simulator is not None:
                    simulator.start()
                    st.session_state["streaming_active"] = True
                    st.rerun()

        with ctrl_col2:
            if st.button("⏸️ Pause", use_container_width=True, disabled=not is_running):
                if simulator is not None:
                    simulator.pause()
                    st.session_state["streaming_active"] = False
                    st.rerun()

        with ctrl_col3:
            if st.button(
                "⏹️ Stop", use_container_width=True, disabled=not (is_running or is_paused)
            ):
                if simulator is not None:
                    simulator.stop()
                    st.session_state["streaming_active"] = False
                    st.rerun()

        with ctrl_col4:
            if st.button("⏭️ Step (1 Txn)", use_container_width=True, disabled=simulator is None):
                if simulator is not None:
                    res = simulator.step()
                    st.rerun()

        with ctrl_col5:
            if st.button("⏩ Step Batch (20)", use_container_width=True, disabled=simulator is None):
                if simulator is not None:
                    simulator.step_batch(20)
                    st.rerun()

        with ctrl_col6:
            if st.button("🔄 Reset Stream", use_container_width=True, disabled=simulator is None):
                if simulator is not None:
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
        if st.session_state.get("streaming_active", False) and is_running and simulator is not None:
            if not getattr(simulator, "is_finished", False):
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
                },
            )
        else:
            st.info("No incoming transactions yet. Click **▶️ Start Stream** or **⏭️ Step** above.")

    # =========================================================================
    # TAB 2: Excel / CSV Batch Predictor
    # =========================================================================
    with tab_excel:
        st.markdown("### 📑 Batch Transaction Scoring from Excel / CSV")
        st.markdown(
            "Upload an Excel spreadsheet (`.xlsx`, `.xls`) or CSV file containing transactions. "
            "The system evaluates each record against the production ML pipeline, generates fraud "
            "probabilities, assigns risk bands, and enables direct ingestion into the live alert queue."
        )

        # Template & Format Guidance
        t_col1, t_col2, t_col3 = st.columns([1, 1, 1.5])
        with t_col1:
            excel_template_bytes = generate_sample_template(simulator=simulator, file_format="xlsx")
            st.download_button(
                label="📥 Download Sample Excel (.xlsx)",
                data=excel_template_bytes,
                file_name="sample_transactions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with t_col2:
            csv_template_bytes = generate_sample_template(simulator=simulator, file_format="csv")
            st.download_button(
                label="📥 Download Sample CSV (.csv)",
                data=csv_template_bytes,
                file_name="sample_transactions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with t_col3:
            with st.expander("ℹ️ Expected Format & Schema Notes", expanded=False):
                st.markdown(
                    """
                    - **Required**: `Amount` (dollar amount of transaction)
                    - **Recommended**: `Time` (seconds from initial transaction; defaults to 0.0 if omitted)
                    - **PCA Features**: `V1` to `V28`. If any PCA columns are omitted, they will default to `0.0` (neutral baseline).
                    - **Optional**: `Class` (ground truth label 0 or 1; automatically stripped before inference to prevent leakage).
                    """
                )

        st.markdown("---")

        # File Uploader
        uploaded_file = st.file_uploader(
            "Upload Transaction File (Excel or CSV)",
            type=["xlsx", "xls", "csv"],
            help="Select an .xlsx, .xls, or .csv spreadsheet file with transaction data.",
        )

        if uploaded_file is not None:
            try:
                # 1. Parse File
                file_ext = uploaded_file.name.split(".")[-1].lower()
                if file_ext in ("xlsx", "xls"):
                    input_df = pd.read_excel(uploaded_file)
                else:
                    input_df = pd.read_csv(uploaded_file)

                st.success(f"Successfully loaded `{uploaded_file.name}` ({len(input_df):,} rows, {len(input_df.columns)} columns).")

                # Preview Raw Input
                with st.expander("🔍 Preview Uploaded Raw Data (First 5 Rows)", expanded=False):
                    st.dataframe(input_df.head(5), use_container_width=True)

                # 2. Schema Validation & Imputation
                # Check Amount
                amount_col_matches = [c for c in input_df.columns if c.lower() == "amount"]
                if amount_col_matches:
                    input_df = input_df.rename(columns={amount_col_matches[0]: AMOUNT_COLUMN})
                elif AMOUNT_COLUMN not in input_df.columns:
                    st.error("❌ Missing required column: `Amount`. Please include a transaction amount column.")
                    return

                # Check Time
                time_col_matches = [c for c in input_df.columns if c.lower() == "time"]
                if time_col_matches:
                    input_df = input_df.rename(columns={time_col_matches[0]: TIME_COLUMN})
                elif TIME_COLUMN not in input_df.columns:
                    input_df[TIME_COLUMN] = 0.0

                # Impute missing PCA features
                missing_pca = [f for f in PCA_FEATURES if f not in input_df.columns]
                if missing_pca:
                    st.info(f"💡 Defaulted {len(missing_pca)} missing PCA features to neutral baseline 0.0.")
                    for col in missing_pca:
                        input_df[col] = 0.0

                # Ensure zero NaN in feature columns
                for col in FEATURE_COLUMNS:
                    if input_df[col].isnull().any():
                        input_df[col] = input_df[col].fillna(0.0)

                # Prepare feature matrix for inference (prevent leakage)
                inference_features = input_df[FEATURE_COLUMNS].copy()

                # 3. Model Inference & Risk Decision Scoring
                with st.spinner("⚡ Running production model inference and risk evaluation..."):
                    probs = predictor.predict_batch_probabilities(inference_features)

                    # Build evaluated records
                    scored_rows = []
                    allow_count = 0
                    review_count = 0
                    block_count = 0
                    total_flagged_amount = 0.0

                    for idx, row in input_df.iterrows():
                        prob = float(probs[idx])
                        amt = float(row[AMOUNT_COLUMN])
                        time_val = float(row[TIME_COLUMN])
                        evaluation = decision_engine.evaluate(prob, amount=amt)

                        risk = evaluation["risk_level"]
                        decision = evaluation["decision"]

                        if decision == "ALLOW":
                            allow_count += 1
                        elif decision == "REVIEW":
                            review_count += 1
                            total_flagged_amount += amt
                        elif decision == "BLOCK":
                            block_count += 1
                            total_flagged_amount += amt

                        # Determine Transaction ID
                        custom_id = str(row.get("Transaction_ID", row.get("ID", f"EXCEL-{idx+1:04d}")))

                        scored_rows.append({
                            "Transaction ID": custom_id,
                            "Time (s)": time_val,
                            "Amount ($)": amt,
                            "Fraud Probability": prob,
                            "Risk Level": risk,
                            "Decision": decision,
                            "Explanation": evaluation["explanation"],
                            "Action Taken": evaluation["action"],
                        })

                scored_df = pd.DataFrame(scored_rows)

                # 4. KPI Metrics Cards
                st.markdown("### 📊 Scoring Overview & Risk Distribution")
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Total Ingested", f"{len(scored_df):,}")
                m2.metric("🟢 ALLOW (Clean)", f"{allow_count:,}", f"{(allow_count / len(scored_df)):.1%}")
                m3.metric("🟡 REVIEW (MFA/Queue)", f"{review_count:,}", f"{(review_count / len(scored_df)):.1%}")
                m4.metric("🔴 BLOCK (Declined)", f"{block_count:,}", f"{(block_count / len(scored_df)):.1%}")
                m5.metric("💰 Amount at Risk", f"${total_flagged_amount:,.2f}")

                # 5. Visual Chart
                chart_col1, chart_col2 = st.columns([1, 1.4])
                with chart_col1:
                    dec_counts = pd.DataFrame({
                        "Decision": ["ALLOW", "REVIEW", "BLOCK"],
                        "Count": [allow_count, review_count, block_count],
                    })
                    fig = px.pie(
                        dec_counts,
                        values="Count",
                        names="Decision",
                        color="Decision",
                        color_discrete_map=DECISION_COLORS,
                        hole=0.45,
                        title="Decision Engine Breakdown",
                    )
                    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=260)
                    st.plotly_chart(fig, use_container_width=True)

                with chart_col2:
                    fig_hist = px.histogram(
                        scored_df,
                        x="Fraud Probability",
                        nbins=30,
                        color="Decision",
                        color_discrete_map=DECISION_COLORS,
                        title="Fraud Probability Distribution",
                        labels={"Fraud Probability": "Estimated Fraud Probability"},
                    )
                    fig_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10), height=260)
                    st.plotly_chart(fig_hist, use_container_width=True)

                # 6. Filterable Results Dataframe
                st.markdown("### 📋 Evaluated Transactions")
                filter_col1, filter_col2 = st.columns([1, 2])
                with filter_col1:
                    decision_filter = st.selectbox(
                        "Filter by Decision",
                        options=["All", "ALLOW", "REVIEW", "BLOCK"],
                        index=0,
                    )
                with filter_col2:
                    min_prob = st.slider("Filter by Minimum Probability (%)", 0, 100, 0, 5)

                filtered_df = scored_df.copy()
                if decision_filter != "All":
                    filtered_df = filtered_df[filtered_df["Decision"] == decision_filter]
                if min_prob > 0:
                    filtered_df = filtered_df[filtered_df["Fraud Probability"] >= (min_prob / 100.0)]

                # Format columns for display
                display_df = filtered_df.copy()
                display_df["Amount ($)"] = display_df["Amount ($)"].map(lambda x: f"${x:,.2f}")
                display_df["Fraud Probability"] = display_df["Fraud Probability"].map(lambda x: f"{x:.2%}")

                st.dataframe(
                    display_df[[
                        "Transaction ID",
                        "Amount ($)",
                        "Fraud Probability",
                        "Risk Level",
                        "Decision",
                        "Explanation",
                    ]],
                    hide_index=True,
                    use_container_width=True,
                )

                # 7. Action Bar: Download & Commit to Live System
                st.markdown("### 🚀 Operational Actions")
                action_col1, action_col2, action_col3 = st.columns([1, 1, 1])

                with action_col1:
                    # Export Scored Excel
                    export_buffer = io.BytesIO()
                    full_export_df = pd.concat([scored_df, input_df[PCA_FEATURES]], axis=1)
                    full_export_df.to_excel(export_buffer, index=False, engine="openpyxl")
                    st.download_button(
                        label="💾 Download Scored Excel (.xlsx)",
                        data=export_buffer.getvalue(),
                        file_name=f"scored_{uploaded_file.name}",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

                with action_col2:
                    # Ingest into Live Memory Store & Alert Queue
                    if st.button("📥 Commit All to Live Store & Alerts Queue", use_container_width=True):
                        ingested_count = 0
                        timestamp_prefix = int(time.time())
                        for idx, item in scored_df.iterrows():
                            # Ensure unique ID in case of repeats
                            txn_id = f"{item['Transaction ID']}-{timestamp_prefix}"
                            try:
                                store.add_transaction(
                                    transaction_id=txn_id,
                                    amount=float(item["Amount ($)"]),
                                    fraud_probability=float(item["Fraud Probability"]),
                                    risk_level=item["Risk Level"],
                                    decision=item["Decision"],
                                    elapsed_time=float(item["Time (s)"]),
                                    metadata={"source": "Excel Upload", "file": uploaded_file.name},
                                )
                                ingested_count += 1
                            except ValueError:
                                continue

                        st.success(
                            f"🎉 Ingested {ingested_count:,} transactions into live memory store! "
                            f"{review_count + block_count:,} alerts queued for fraud investigation on the Alerts page."
                        )
                        st.rerun()

                with action_col3:
                    # Load into Simulator as live replay queue
                    if st.button("🔄 Set as Live Simulator Stream", use_container_width=True):
                        # Reconstruct full dataset for simulator replay
                        full_stream_df = pd.concat([input_df[FEATURE_COLUMNS]], axis=1)
                        if TARGET_COLUMN not in full_stream_df.columns:
                            # Assign dummy class 0 if not present
                            full_stream_df[TARGET_COLUMN] = 0
                        simulator._test_df = full_stream_df.reset_index(drop=True)
                        simulator.reset()
                        st.success(
                            f"✅ Simulator loaded with {len(full_stream_df):,} records from `{uploaded_file.name}`. "
                            "Switch to the **📡 Live Replay Simulator** tab to stream them!"
                        )

            except Exception as err:
                st.error(f"❌ Error processing spreadsheet: {err}")


if __name__ == "__main__":
    from pathlib import Path
    import sys
    ROOT = Path(__file__).resolve().parent.parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from app.streamlit_app import initialize_session_state, LOGO_PATH
    if LOGO_PATH.is_file():
        st.logo(str(LOGO_PATH), size="large")
    initialize_session_state()
    render_live_monitoring(
        st.session_state["transaction_store"],
        st.session_state["decision_engine"],
        st.session_state["simulator"],
    )
