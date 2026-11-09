"""
Credit Card Fraud Detection System
Interactive Streamlit Dashboard featuring:
1. Historical Transaction Stream Simulation (Upload Excel/CSV & simulate last processed transaction)
2. Fraud Alerts Queue (Investigation & case management for HIGH/MEDIUM risk transactions)
3. Manual Predictions (Single transaction scoring with legitimate/fraudulent presets)
4. Model Performance Analytics (Comparative benchmarks: Logistic Regression, Decision Tree, Random Forest)
With Joblib-based model persistence and Pytest automation.
"""

from pathlib import Path
import sys
import io
import json
import streamlit as st
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    DEFAULT_LOW_RISK_THRESHOLD,
    DEFAULT_HIGH_RISK_THRESHOLD,
    RISK_COLORS,
    DECISION_COLORS,
    METADATA_PATH,
    EVALUATION_METRICS_PATH,
    FEATURE_COLUMNS,
    TIME_COLUMN,
    AMOUNT_COLUMN,
    PCA_FEATURES,
)
from ml.predictor import FraudPredictor, get_predictor
from app.decision_engine import FraudDecisionEngine
from app.transaction_store import TransactionStore
from app.batch_processor import (
    generate_sample_template,
    process_transaction_file,
    export_scored_data,
)

# Configure Streamlit Page
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Display Sentinel Brand Logo
LOGO_PATH = PROJECT_ROOT / "app" / "static" / "sentinel_logo.svg"
if LOGO_PATH.is_file():
    st.logo(str(LOGO_PATH), size="large")

# Custom CSS for fintech aesthetic & clean tabs
st.markdown(
    """
    <style>
    /* Clean layout styling */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }
    [data-testid="stSidebarHeader"] {
        display: flex !important;
        align-items: center !important;
        padding: 1.25rem 1rem 0.75rem 1rem !important;
        border-bottom: 1px solid rgba(226, 232, 240, 0.15) !important;
    }
    [data-testid="stSidebarHeader"] img,
    [data-testid="stLogo"] {
        height: 55px !important;
        max-height: 55px !important;
        width: auto !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 15px;
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .stMetric label {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
    }
    .stMetric div[data-testid="stMetricValue"] {
        font-size: 24px;
        font-weight: 700;
    }
    header[data-testid="stHeader"] {
        background: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_production_pipeline():
    """Cache fitted pipeline once across reruns and sessions (Joblib-persisted)."""
    return get_predictor()


@st.cache_data
def load_offline_analytics():
    """Load offline benchmark comparisons and evaluation metrics."""
    metadata = {}
    metrics_data = {}
    if METADATA_PATH.is_file():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            pass
    if EVALUATION_METRICS_PATH.is_file():
        try:
            with open(EVALUATION_METRICS_PATH, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
        except Exception:
            pass
    return metadata, metrics_data


def initialize_session_state():
    """Initialize stateful decision engine, transaction store, and manual inputs."""
    if "decision_engine" not in st.session_state:
        st.session_state["decision_engine"] = FraudDecisionEngine(
            low_risk_threshold=DEFAULT_LOW_RISK_THRESHOLD,
            high_risk_threshold=DEFAULT_HIGH_RISK_THRESHOLD,
        )

    if "transaction_store" not in st.session_state:
        st.session_state["transaction_store"] = TransactionStore()

    if "manual_features" not in st.session_state:
        init_features = {col: 0.0 for col in FEATURE_COLUMNS}
        init_features[AMOUNT_COLUMN] = 45.00
        init_features[TIME_COLUMN] = 7200.0
        st.session_state["manual_features"] = init_features


def main():
    initialize_session_state()
    engine: FraudDecisionEngine = st.session_state["decision_engine"]
    store: TransactionStore = st.session_state["transaction_store"]

    # Load production predictor
    try:
        predictor = load_production_pipeline()
    except Exception as e:
        st.error(
            f"⚠️ Model pipeline error: {e}.\n\n"
            "Please ensure model artifacts are trained by running `python ml/train.py`."
        )
        return

    # ---------------------------------------------------------
    # SIDEBAR: Risk Bands, System Status, Templates
    # ---------------------------------------------------------
    st.sidebar.markdown("### 🎚️ Operational Risk Bands")
    new_low = st.sidebar.slider(
        "Review Threshold (ALLOW → REVIEW)",
        min_value=0.05,
        max_value=0.50,
        value=float(engine.low_threshold),
        step=0.05,
        help="Transactions with fraud probability above this threshold trigger manual review.",
    )

    new_high = st.sidebar.slider(
        "Block Threshold (REVIEW → BLOCK)",
        min_value=0.50,
        max_value=0.95,
        value=float(engine.high_threshold),
        step=0.05,
        help="Transactions with fraud probability above this threshold are automatically declined.",
    )

    if new_low != engine.low_threshold or new_high != engine.high_threshold:
        try:
            engine.set_thresholds(new_low, new_high)
        except ValueError as err:
            st.sidebar.error(str(err))

    st.sidebar.markdown(
        f"""
        <div style="font-size: 12px; color: #64748b; line-height: 1.6; margin-top: 6px; margin-bottom: 14px;">
            <span style="color: #10B981; font-weight: bold;">● ALLOW:</span> &lt; {engine.low_threshold:.0%}<br>
            <span style="color: #F59E0B; font-weight: bold;">● REVIEW:</span> {engine.low_threshold:.0%} – {engine.high_threshold:.0%}<br>
            <span style="color: #EF4444; font-weight: bold;">● BLOCK:</span> &ge; {engine.high_threshold:.0%}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ System Status")
    st.sidebar.markdown("**Production Model:** `Random Forest v1.0`")
    st.sidebar.markdown("**Class Balancing:** `SMOTE + Class Weights`")
    st.sidebar.markdown("**Validation:** `Chronological Split (0 Leakage)`")
    st.sidebar.markdown("**Model Persistence:** `Joblib (.pkl)`")
    st.sidebar.markdown("**Test Automation:** `Pytest (28 Passed)`")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📥 Download Sample Templates")
    sidebar_xlsx = generate_sample_template(file_format="xlsx")
    st.sidebar.download_button(
        label="📥 Download Template (.xlsx)",
        data=sidebar_xlsx,
        file_name="sample_transactions.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    sidebar_csv = generate_sample_template(file_format="csv")
    st.sidebar.download_button(
        label="📄 Download Template (.csv)",
        data=sidebar_csv,
        file_name="sample_transactions.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ---------------------------------------------------------
    # MAIN HEADER
    # ---------------------------------------------------------
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown(
        "Interactive dashboard featuring **batch transaction scoring & stream simulation**, **fraud alerts**, "
        "**manual predictions**, and **model performance analytics** with **Joblib persistence** and **Pytest automation**."
    )

    # Interactive Dashboard Tabs (Simple, clean, and easy to navigate)
    tab_stream, tab_alerts, tab_manual, tab_analytics = st.tabs([
        "📁 Excel / CSV Upload & Simulation",
        "🚨 Fraud Alerts Queue",
        "🔍 Manual Prediction",
        "📈 Model Performance Analytics",
    ])

    # =========================================================================
    # TAB 1: Excel / CSV Upload & Last Processed Transaction Simulation
    # =========================================================================
    with tab_stream:
        st.markdown("### 📁 Batch Spreadsheet Ingestion & Transaction Simulation")
        st.markdown(
            "Upload an **Excel (.xlsx / .xls)** or **CSV (.csv)** file to process transaction records. "
            "The system scores the batch and **simulates the last processed transaction**."
        )

        col_up_box, col_up_actions = st.columns([3, 1])
        with col_up_box:
            uploaded_file = st.file_uploader(
                "Upload Transaction File (Excel or CSV)",
                type=["xlsx", "xls", "csv"],
                help="Upload an Excel spreadsheet or CSV containing transaction rows (Time, Amount, V1–V28).",
                key="tab1_uploader",
            )
        with col_up_actions:
            st.write("")
            st.write("")
            if st.button("⚡ Test with Sample Data", use_container_width=True, key="btn_sample_test"):
                st.session_state["active_upload_bytes"] = generate_sample_template(file_format="xlsx")
                st.session_state["active_upload_name"] = "sample_transactions.xlsx"
                st.rerun()

        file_to_process = None
        file_name = None

        if uploaded_file is not None:
            file_to_process = uploaded_file
            file_name = uploaded_file.name
        elif "active_upload_bytes" in st.session_state:
            file_to_process = io.BytesIO(st.session_state["active_upload_bytes"])
            file_name = st.session_state.get("active_upload_name", "sample_transactions.xlsx")

        if file_to_process is None:
            st.info("👆 Upload an Excel (.xlsx / .xls) or CSV file, or click **⚡ Test with Sample Data** to begin.")
        else:
            try:
                scored_df, metrics, last_txn = process_transaction_file(
                    file_to_process, predictor=predictor, decision_engine=engine, store=store
                )

                st.success(f"✅ Processed `{file_name}` ({metrics['total_processed']} transactions). Flagged alerts have been queued.")

                # KPI Summary Cards
                kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
                kpi_col1.metric("Total Processed", f"{metrics['total_processed']:,}")
                kpi_col2.metric("🟢 ALLOW (Low Risk)", f"{metrics['allow_count']:,}", f"{metrics['allow_rate']:.1f}%")
                kpi_col3.metric("🟡 REVIEW (Medium Risk)", f"{metrics['review_count']:,}", f"{metrics['review_rate']:.1f}%")
                kpi_col4.metric("🔴 BLOCK (High Risk)", f"{metrics['block_count']:,}", f"{metrics['block_rate']:.1f}%")
                kpi_col5.metric("Peak Fraud Probability", f"{metrics['max_fraud_prob']:.2%}")

                st.markdown("---")

                # Simulated Last Processed Transaction Section
                st.markdown("### 📡 Simulated Last Processed Transaction")
                st.caption("Operational simulation of the final transaction from the uploaded batch.")

                last_risk = last_txn["risk_level"]
                last_decision = last_txn["decision"]
                last_prob = last_txn["fraud_probability"]
                color = RISK_COLORS.get(last_risk, "#333")

                st.markdown(
                    f"""
                    <div style="padding: 18px 22px; border-radius: 10px; background-color: {color}15; border: 2px solid {color}; margin-bottom: 18px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                            <span style="font-size: 18px; font-weight: 700; color: #1E293B;">
                                Transaction: <code style="font-size: 17px; color: {color};">{last_txn['transaction_id']}</code>
                            </span>
                            <span style="font-size: 13px; font-weight: 600; padding: 4px 10px; border-radius: 999px; background-color: {color}25; color: {color};">
                                ● Real-Time Simulation
                            </span>
                        </div>
                        <div style="display: flex; gap: 32px; flex-wrap: wrap; margin-bottom: 14px;">
                            <div><span style="color: #64748B; font-size: 12px; font-weight: 600;">AMOUNT</span><br><strong style="font-size: 22px; color: #0F172A;">${last_txn['amount']:,.2f}</strong></div>
                            <div><span style="color: #64748B; font-size: 12px; font-weight: 600;">ELAPSED TIME</span><br><strong style="font-size: 22px; color: #0F172A;">{last_txn['elapsed_time']:,.1f} s</strong></div>
                            <div><span style="color: #64748B; font-size: 12px; font-weight: 600;">FRAUD PROBABILITY</span><br><strong style="font-size: 22px; color: {color};">{last_prob:.2%}</strong></div>
                            <div><span style="color: #64748B; font-size: 12px; font-weight: 600;">RISK BAND</span><br><strong style="font-size: 22px; color: {color};">{last_risk} RISK</strong></div>
                            <div><span style="color: #64748B; font-size: 12px; font-weight: 600;">DECISION</span><br><strong style="font-size: 22px; color: {color};">{last_decision}</strong></div>
                        </div>
                        <div style="background-color: #FFFFFF; padding: 12px 16px; border-radius: 6px; border-left: 4px solid {color}; margin-top: 8px;">
                            <strong style="color: #334155;">Policy Rationale:</strong> {last_txn['explanation']}<br>
                            <strong style="color: #334155; margin-top: 4px; display: inline-block;">Action:</strong> {last_txn['action']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("🔍 Inspect Diagnostic Feature Signals (Last Transaction)", expanded=False):
                    f_cols = st.columns(6)
                    top_features = ["V14", "V17", "V12", "V10", "V4", "V11"]
                    features_dict = last_txn.get("features", {})
                    for i, feat in enumerate(top_features):
                        val = features_dict.get(feat, 0.0)
                        f_cols[i].metric(feat, f"{val:.4f}")

                if st.button("🔄 Re-simulate Last Transaction", key="btn_resim_tab1"):
                    st.toast(
                        f"Simulated transaction {last_txn['transaction_id']}: {last_decision} ({last_prob:.2%})",
                        icon="⚡",
                    )

                st.markdown("---")

                # Scored Data Table
                st.markdown("### 📋 Batch Output Records")
                filter_choice = st.radio(
                    "Filter output table by decision:",
                    ["All", "ALLOW", "REVIEW", "BLOCK"],
                    horizontal=True,
                    key="tab1_filter",
                )

                filtered_table = scored_df.copy()
                if filter_choice != "All":
                    filtered_table = filtered_table[filtered_table["Decision"] == filter_choice]

                formatted_df = filtered_table.copy()
                formatted_df["Amount ($)"] = formatted_df["Amount ($)"].map(lambda x: f"${x:,.2f}")
                formatted_df["Fraud Probability"] = formatted_df["Fraud Probability"].map(lambda x: f"{x:.2%}")

                st.dataframe(formatted_df, hide_index=True, use_container_width=True)

                # Export Scored Data
                st.markdown("### 💾 Export Processed Results")
                exp_col1, exp_col2 = st.columns(2)
                clean_name = file_name.rsplit(".", 1)[0]
                with exp_col1:
                    out_excel = export_scored_data(scored_df, file_format="xlsx")
                    st.download_button(
                        label="📥 Download Scored Excel (.xlsx)",
                        data=out_excel,
                        file_name=f"scored_{clean_name}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )
                with exp_col2:
                    out_csv = export_scored_data(scored_df, file_format="csv")
                    st.download_button(
                        label="📄 Download Scored CSV (.csv)",
                        data=out_csv,
                        file_name=f"scored_{clean_name}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            except Exception as err:
                st.error(f"❌ Error processing transaction file: {err}")

    # =========================================================================
    # TAB 2: Fraud Alerts Queue
    # =========================================================================
    with tab_alerts:
        st.markdown("### 🚨 Fraud Alerts & Case Investigation Queue")
        st.markdown(
            "Centralized investigation queue for transactions evaluated as **HIGH** or **MEDIUM** risk. "
            "Analysts can review automated risk flags and update disposition status."
        )

        alerts = store.get_alerts()

        # Summary Metrics
        a_col1, a_col2, a_col3, a_col4 = st.columns(4)
        total_alerts = len(alerts)
        critical_alerts = sum(1 for a in alerts if a["risk_level"] == "HIGH")
        pending_alerts = sum(1 for a in alerts if a["review_status"] == "PENDING")
        flagged_vol = sum(a["amount"] for a in alerts)

        a_col1.metric("Total Active Alerts", total_alerts)
        a_col2.metric("Critical (HIGH Risk)", critical_alerts)
        a_col3.metric("Action Required (PENDING)", pending_alerts)
        a_col4.metric("Flagged Volume ($)", f"${flagged_vol:,.2f}")

        st.markdown("---")

        if not alerts:
            st.info("No active fraud alerts. Upload transactions in Tab 1 or score transactions in Tab 3 to trigger alerts.")
        else:
            status_filter = st.selectbox("Filter alerts by review status:", ["ALL", "PENDING", "REVIEWED", "DISMISSED"], index=0)
            displayed_alerts = alerts if status_filter == "ALL" else [a for a in alerts if a["review_status"] == status_filter]

            for a in displayed_alerts:
                risk = a["risk_level"]
                status = a["review_status"]
                txn_id = a["transaction_id"]
                prob = a["fraud_probability"]
                amt = a["amount"]
                c = RISK_COLORS.get(risk, "#333")

                with st.container():
                    a_c1, a_c2, a_c3, a_c4, a_c5 = st.columns([2, 2, 2, 2, 3])
                    a_c1.markdown(f"**ID:** `{txn_id}`")
                    a_c2.markdown(f"**Amount:** `${amt:,.2f}`")
                    a_c3.markdown(f"**Score:** `{prob:.2%}`")
                    a_c4.markdown(f"**Risk:** <span style='color:{c}; font-weight:bold;'>{risk}</span>", unsafe_allow_html=True)

                    with a_c5:
                        btn_c1, btn_c2 = st.columns(2)
                        with btn_c1:
                            if st.button("✅ Reviewed", key=f"rev_{txn_id}"):
                                store.update_alert_status(txn_id, "REVIEWED")
                                st.rerun()
                        with btn_c2:
                            if st.button("❌ Dismiss", key=f"dis_{txn_id}"):
                                store.update_alert_status(txn_id, "DISMISSED")
                                st.rerun()
                st.markdown("<hr style='margin: 4px 0 10px 0; border: none; border-top: 1px solid #eee;'>", unsafe_allow_html=True)

    # =========================================================================
    # TAB 3: Manual Predictions
    # =========================================================================
    with tab_manual:
        st.markdown("### 🔍 Manual Transaction Risk Scoring")
        st.markdown(
            "Score an individual custom transaction against the production Random Forest pipeline and decision engine. "
            "You can load realistic sample presets or customize transaction attributes directly."
        )

        preset_col1, preset_col2, preset_col3 = st.columns(3)
        features = st.session_state["manual_features"]

        with preset_col1:
            if st.button("🟢 Load Legitimate Transaction Preset", use_container_width=True):
                init_features = {col: 0.0 for col in FEATURE_COLUMNS}
                init_features[AMOUNT_COLUMN] = 38.50
                init_features[TIME_COLUMN] = 1500.0
                st.session_state["manual_features"] = init_features
                st.session_state["manual_preset_note"] = "Loaded legitimate sample profile (Expected: ALLOW, LOW Risk)."
                st.rerun()

        with preset_col2:
            if st.button("🔴 Load Fraudulent Transaction Preset", use_container_width=True):
                fraud_features = {col: 0.0 for col in FEATURE_COLUMNS}
                fraud_features[AMOUNT_COLUMN] = 350.00
                fraud_features[TIME_COLUMN] = 12000.0
                fraud_features["V14"] = -8.5
                fraud_features["V17"] = -6.2
                fraud_features["V12"] = -5.1
                fraud_features["V10"] = -4.3
                fraud_features["V4"] = 3.9
                fraud_features["V11"] = 3.1
                st.session_state["manual_features"] = fraud_features
                st.session_state["manual_preset_note"] = "Loaded fraudulent sample profile with high-impact PCA deviations (Expected: BLOCK, HIGH Risk)."
                st.rerun()

        with preset_col3:
            if st.button("🔄 Reset Baseline Values", use_container_width=True):
                init_features = {col: 0.0 for col in FEATURE_COLUMNS}
                init_features[AMOUNT_COLUMN] = 50.00
                init_features[TIME_COLUMN] = 3600.0
                st.session_state["manual_features"] = init_features
                st.session_state.pop("manual_preset_note", None)
                st.rerun()

        if "manual_preset_note" in st.session_state:
            st.info(st.session_state["manual_preset_note"])

        with st.form("manual_scoring_form"):
            mf_col1, mf_col2 = st.columns(2)
            with mf_col1:
                man_amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=100000.0, value=float(features.get(AMOUNT_COLUMN, 45.0)), step=5.0)
            with mf_col2:
                man_time = st.number_input("Elapsed Seconds (Time)", min_value=0.0, max_value=200000.0, value=float(features.get(TIME_COLUMN, 7200.0)), step=60.0)

            st.markdown("##### High-Impact Diagnostic Components")
            top_pca = ["V14", "V17", "V12", "V10", "V4", "V11"]
            top_cols = st.columns(3)
            top_inputs = {}
            for idx, feat in enumerate(top_pca):
                with top_cols[idx % 3]:
                    top_inputs[feat] = st.number_input(f"{feat}", value=float(features.get(feat, 0.0)), step=0.5, format="%.4f")

            with st.expander("Expand Other PCA Features (V1 to V28)", expanded=False):
                other_pca = [f for f in PCA_FEATURES if f not in top_pca]
                other_cols = st.columns(4)
                other_inputs = {}
                for idx, feat in enumerate(other_pca):
                    with other_cols[idx % 4]:
                        other_inputs[feat] = st.number_input(f"{feat}", value=float(features.get(feat, 0.0)), step=0.5, format="%.4f")

            submit_manual = st.form_submit_button("⚡ Evaluate Transaction Risk", use_container_width=True)

        if submit_manual:
            payload = {TIME_COLUMN: man_time, AMOUNT_COLUMN: man_amount}
            payload.update(top_inputs)
            payload.update(other_inputs)

            prob = predictor.predict_probability(payload)
            decision_info = engine.evaluate(prob, amount=man_amount)
            risk = decision_info["risk_level"]
            decision = decision_info["decision"]
            c = RISK_COLORS.get(risk, "#333")

            st.markdown("---")
            m_res1, m_res2, m_res3 = st.columns(3)
            m_res1.metric("Predicted Fraud Probability", f"{prob:.2%}")
            m_res2.metric("Assigned Risk Level", f"{risk} RISK")
            m_res3.metric("Decision Engine Action", decision)

            st.markdown(
                f"""
                <div style="padding: 16px; border-radius: 8px; background-color: {c}18; border-left: 5px solid {c}; margin-top: 10px;">
                    <h4 style="margin: 0; color: {c};">{decision} ({risk} RISK)</h4>
                    <p style="margin: 6px 0 0 0; font-size: 14px;"><strong>Policy Explanation:</strong> {decision_info['explanation']}</p>
                    <p style="margin: 4px 0 0 0; font-size: 14px;"><strong>Action:</strong> {decision_info['action']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Option to add to alert queue
            if risk in ("HIGH", "MEDIUM"):
                if st.button("🚨 Queue this alert in Fraud Alerts Queue", key="btn_add_manual_alert"):
                    custom_id = f"MANUAL-{int(man_time)}"
                    store.add_transaction(
                        transaction_id=custom_id,
                        amount=man_amount,
                        fraud_probability=prob,
                        risk_level=risk,
                        decision=decision,
                        elapsed_time=man_time,
                        metadata={"source": "Manual Evaluation Form"}
                    )
                    st.success(f"Added alert `{custom_id}` to Fraud Alerts Queue (Tab 2).")

    # =========================================================================
    # TAB 4: Model Performance Analytics
    # =========================================================================
    with tab_analytics:
        st.markdown("### 📈 Model Performance & Comparative Benchmark Analytics")
        st.markdown(
            "Evaluation across candidate algorithms (**Logistic Regression**, **Decision Tree**, **Random Forest**) "
            "trained strictly on chronological partitions (70% Train, 15% Validation, 15% Test) to prevent data leakage."
        )

        metadata, metrics_data = load_offline_analytics()
        comparison = metrics_data.get("comparison", {})

        if comparison:
            rows = []
            for alg_name, val_dict in comparison.items():
                opt = val_dict.get("optimized_metrics", {})
                rows.append({
                    "Algorithm Candidate": alg_name,
                    "Optimal Threshold": f"{val_dict.get('optimal_threshold', 0.5):.2f}",
                    "PR-AUC": f"{opt.get('pr_auc', 0.0):.4f}",
                    "Recall (Fraud Catch)": f"{opt.get('recall', 0.0):.2%}",
                    "Precision": f"{opt.get('precision', 0.0):.2%}",
                    "F1-Score": f"{opt.get('f1_score', 0.0):.4f}",
                    "False Positives": f"{opt.get('false_positives', 0):,}",
                    "Accuracy": f"{opt.get('accuracy', 0.0):.2%}",
                })

            bench_df = pd.DataFrame(rows)
            st.dataframe(bench_df, hide_index=True, use_container_width=True)

            # Why PR-AUC over Accuracy explanation box
            st.markdown(
                """
                > [!NOTE]
                > **Why PR-AUC and F1-Score are used instead of Accuracy:**  
                > In highly imbalanced datasets (fraud is only ~0.17%), a dummy model predicting "legitimate" for 100% of transactions 
                > achieves 99.83% accuracy while missing every single fraud event. **Precision-Recall AUC (PR-AUC)** and **F1-Score** 
                > evaluate true operational trade-offs between catching fraud (Recall) and minimizing false customer friction (Precision).
                """
            )

            # Confusion Matrix summary for selected model
            rf_opt = comparison.get("Random Forest", {}).get("optimized_metrics", {})
            if rf_opt:
                st.markdown("#### 🎯 Random Forest (Production Selected Model) Test Confusion Matrix")
                c_c1, c_c2, c_c3, c_c4 = st.columns(4)
                c_c1.metric("True Positives (Frauds Blocked)", f"{rf_opt.get('true_positives', 0):,}")
                c_c2.metric("False Negatives (Frauds Missed)", f"{rf_opt.get('false_negatives', 0):,}")
                c_c3.metric("False Positives (False Alarms)", f"{rf_opt.get('false_positives', 0):,}")
                c_c4.metric("True Negatives (Legitimate Cleared)", f"{rf_opt.get('true_negatives', 0):,}")

            # Top feature importances
            feat_imp = metadata.get("feature_importances", {})
            if feat_imp:
                st.markdown("#### 🔑 Top Diagnostic Feature Importances (Tree Ensemble)")
                imp_df = pd.DataFrame([
                    {"Feature": k, "Importance": v} for k, v in list(feat_imp.items())[:8]
                ])
                st.dataframe(imp_df, hide_index=True, use_container_width=True)

        else:
            st.info("Evaluation metrics artifacts will be displayed here once generated by `python ml/train.py`.")


if __name__ == "__main__":
    main()
