"""
Page 2: Live Transaction Monitoring and Batch File Scoring.
Provides an Excel/CSV spreadsheet ingestion pipeline with automated feature validation,
batch risk scoring, output display, and simulation of the last processed transaction.
"""

from pathlib import Path
import sys
from typing import Optional
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    AMOUNT_COLUMN,
    DECISION_COLORS,
    FEATURE_COLUMNS,
    PCA_FEATURES,
    RISK_COLORS,
    TIME_COLUMN,
)
from ml.predictor import FraudPredictor, get_predictor
from app.decision_engine import FraudDecisionEngine
from app.batch_processor import (
    generate_sample_template,
    process_transaction_file,
    export_scored_data,
)


def render_live_monitoring(store=None, decision_engine: Optional[FraudDecisionEngine] = None, simulator=None, predictor: Optional[FraudPredictor] = None):
    st.markdown("## 📡 Live Transaction Monitoring & Batch File Scoring")
    st.markdown(
        "Upload an Excel (`.xlsx`, `.xls`) or CSV (`.csv`) file to evaluate transactions against "
        "the production fraud detection pipeline, view batch scoring outputs, and simulate the last processed transaction."
    )

    if predictor is None:
        predictor = getattr(simulator, "predictor", None) or get_predictor()

    if decision_engine is None:
        decision_engine = getattr(simulator, "decision_engine", None) or FraudDecisionEngine()

    # 1. Download Templates & Quick Sample
    col_t1, col_t2, col_t3 = st.columns([1, 1, 1])
    with col_t1:
        xlsx_template = generate_sample_template(simulator=simulator, file_format="xlsx")
        st.download_button(
            label="📥 Download Sample Excel (.xlsx)",
            data=xlsx_template,
            file_name="sample_transactions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with col_t2:
        csv_template = generate_sample_template(simulator=simulator, file_format="csv")
        st.download_button(
            label="📄 Download Sample CSV (.csv)",
            data=csv_template,
            file_name="sample_transactions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_t3:
        if st.button("⚡ Quick Test with Sample Data", use_container_width=True):
            st.session_state["active_upload_bytes"] = generate_sample_template(simulator=simulator, file_format="xlsx")
            st.session_state["active_upload_name"] = "sample_transactions.xlsx"
            st.rerun()

    # 2. File Upload Box
    uploaded_file = st.file_uploader(
        "Upload Transaction File (Excel or CSV)",
        type=["csv", "xlsx", "xls"],
        help="Upload an Excel spreadsheet or CSV containing transaction rows (Time, Amount, V1–V28).",
    )

    file_to_process = None
    file_name = None

    if uploaded_file is not None:
        file_to_process = uploaded_file
        file_name = uploaded_file.name
    elif "active_upload_bytes" in st.session_state:
        import io
        file_to_process = io.BytesIO(st.session_state["active_upload_bytes"])
        file_name = st.session_state.get("active_upload_name", "sample_transactions.xlsx")

    if file_to_process is None:
        st.info("👆 Please upload an Excel (.xlsx / .xls) or CSV file above, or click **⚡ Quick Test with Sample Data** to begin.")
        return

    # 3. Process File and Render Output
    try:
        scored_df, metrics, last_txn = process_transaction_file(
            file_to_process, predictor=predictor, decision_engine=decision_engine
        )

        st.success(f"Successfully processed `{file_name}` ({metrics['total_processed']} transactions).")

        # Summary KPIs
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Total Processed", f"{metrics['total_processed']:,}")
        m_col2.metric("🟢 ALLOW (Low Risk)", f"{metrics['allow_count']:,}", f"{metrics['allow_rate']:.1f}%")
        m_col3.metric("🟡 REVIEW (Medium Risk)", f"{metrics['review_count']:,}", f"{metrics['review_rate']:.1f}%")
        m_col4.metric("🔴 BLOCK (High Risk)", f"{metrics['block_count']:,}", f"{metrics['block_rate']:.1f}%")
        m_col5.metric("Peak Fraud Score", f"{metrics['max_fraud_prob']:.2%}")

        st.markdown("---")

        # 4. Simulated Last Processed Transaction Section
        st.markdown("### 📡 Simulated Last Processed Transaction")
        st.caption("Real-time operational simulation of the final transaction from the processed batch.")

        risk = last_txn["risk_level"]
        decision = last_txn["decision"]
        prob = last_txn["fraud_probability"]
        color = RISK_COLORS.get(risk, "#333")

        st.markdown(
            f"""
            <div style="padding: 18px 24px; border-radius: 10px; background-color: {color}15; border: 2px solid {color}; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 18px; font-weight: 700; color: #1E293B;">
                        Transaction: <code style="font-size: 17px; color: {color};">{last_txn['transaction_id']}</code>
                    </span>
                    <span style="font-size: 13px; font-weight: 600; padding: 4px 10px; border-radius: 999px; background-color: {color}25; color: {color};">
                        ● Real-Time Simulation Active
                    </span>
                </div>
                <div style="display: flex; gap: 30px; flex-wrap: wrap; margin-bottom: 12px;">
                    <div><span style="color: #64748B; font-size: 12px; font-weight: 600;">AMOUNT</span><br><strong style="font-size: 20px; color: #0F172A;">${last_txn['amount']:,.2f}</strong></div>
                    <div><span style="color: #64748B; font-size: 12px; font-weight: 600;">ELAPSED TIME</span><br><strong style="font-size: 20px; color: #0F172A;">{last_txn['elapsed_time']:,.1f} s</strong></div>
                    <div><span style="color: #64748B; font-size: 12px; font-weight: 600;">FRAUD PROBABILITY</span><br><strong style="font-size: 20px; color: {color};">{prob:.2%}</strong></div>
                    <div><span style="color: #64748B; font-size: 12px; font-weight: 600;">ASSIGNED RISK BAND</span><br><strong style="font-size: 20px; color: {color};">{risk} RISK</strong></div>
                    <div><span style="color: #64748B; font-size: 12px; font-weight: 600;">SYSTEM DECISION</span><br><strong style="font-size: 20px; color: {color};">{decision}</strong></div>
                </div>
                <div style="background-color: #FFFFFF; padding: 12px 16px; border-radius: 6px; border-left: 4px solid {color}; margin-top: 10px;">
                    <strong style="color: #334155;">Decision Engine Rationale:</strong> {last_txn['explanation']}<br>
                    <strong style="color: #334155; margin-top: 4px; display: inline-block;">Operational Policy:</strong> {last_txn['action']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("🔍 Inspect Diagnostic Feature Signals of Last Transaction", expanded=False):
            f_cols = st.columns(6)
            top_features = ["V14", "V17", "V12", "V10", "V4", "V11"]
            features_dict = last_txn.get("features", {})
            for i, feat in enumerate(top_features):
                val = features_dict.get(feat, 0.0)
                f_cols[i].metric(feat, f"{val:.4f}")

        if st.button("🔄 Re-simulate Last Transaction"):
            st.toast(f"Simulated transaction {last_txn['transaction_id']}: Decision {decision} ({prob:.2%})", icon="⚡")

        st.markdown("---")

        # 5. Full Processed Output Table
        st.markdown("### 📋 Processed Batch Transactions")

        filter_choice = st.radio(
            "Filter output by decision:",
            ["All", "ALLOW", "REVIEW", "BLOCK"],
            horizontal=True,
        )

        display_df = scored_df.copy()
        if filter_choice != "All":
            display_df = display_df[display_df["Decision"] == filter_choice]

        # Formatting table columns
        formatted_table = display_df.copy()
        formatted_table["Amount ($)"] = formatted_table["Amount ($)"].map(lambda x: f"${x:,.2f}")
        formatted_table["Fraud Probability"] = formatted_table["Fraud Probability"].map(lambda x: f"{x:.2%}")

        st.dataframe(
            formatted_table,
            hide_index=True,
            use_container_width=True,
        )

        # 6. Download Scored Output
        st.markdown("### 💾 Export Processed Results")
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            excel_bytes = export_scored_data(scored_df, file_format="xlsx")
            st.download_button(
                label="📥 Download Scored Results (.xlsx)",
                data=excel_bytes,
                file_name=f"scored_{file_name.replace('.csv', '')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with d_col2:
            csv_bytes = export_scored_data(scored_df, file_format="csv")
            st.download_button(
                label="📄 Download Scored Results (.csv)",
                data=csv_bytes,
                file_name=f"scored_{file_name.replace('.xlsx', '')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    except Exception as err:
        st.error(f"❌ Error processing transaction spreadsheet: {err}")


if __name__ == "__main__":
    from app.streamlit_app import initialize_session_state, LOGO_PATH
    if LOGO_PATH.is_file():
        st.logo(str(LOGO_PATH), size="large")
    initialize_session_state()
    render_live_monitoring(
        st.session_state["transaction_store"],
        st.session_state["decision_engine"],
        st.session_state["simulator"],
    )
