"""
Main Streamlit Application Entrypoint.
Orchestrates sidebar navigation, session state caching, configurable operational thresholds,
and renders modular application pages.
"""

from pathlib import Path
import sys
import streamlit as st

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    DEFAULT_LOW_RISK_THRESHOLD,
    DEFAULT_HIGH_RISK_THRESHOLD,
    MODEL_PATH,
    DATASET_PATH,
)
from ml.predictor import FraudPredictor, get_predictor
from app.decision_engine import FraudDecisionEngine
from app.transaction_store import TransactionStore
from app.simulator import TransactionSimulator

from app.pages.dashboard import render_dashboard
from app.pages.live_monitoring import render_live_monitoring
from app.pages.fraud_alerts import render_fraud_alerts
from app.pages.manual_prediction import render_manual_prediction
from app.pages.model_analytics import render_model_analytics
from app.pages.dataset_explorer import render_dataset_explorer

# Configure Streamlit Page
st.set_page_config(
    page_title="Real-Time Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Display SentinelFraud AI Brand Header at the very top of sidebar (above multipage sections)
LOGO_PATH = PROJECT_ROOT / "app" / "static" / "sentinel_logo.svg"
if LOGO_PATH.is_file():
    st.logo(str(LOGO_PATH), size="large")

# Custom CSS for fintech aesthetic and sidebar branding
st.markdown(
    """
    <style>
    /* Top Sidebar Header & Brand Logo */
    [data-testid="stSidebarHeader"] {
        display: flex !important;
        align-items: center !important;
        padding: 1.25rem 1rem 0.75rem 1rem !important;
        border-bottom: 1px solid rgba(226, 232, 240, 0.15) !important;
    }
    [data-testid="stSidebarHeader"] img,
    [data-testid="stLogo"] {
        height: 60px !important;
        max-height: 60px !important;
        width: auto !important;
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
    """Cache fitted pipeline once across reruns and sessions."""
    return get_predictor()


def initialize_session_state():
    """Initialize stateful instances in Streamlit session state."""
    if "decision_engine" not in st.session_state:
        st.session_state["decision_engine"] = FraudDecisionEngine(
            low_risk_threshold=DEFAULT_LOW_RISK_THRESHOLD,
            high_risk_threshold=DEFAULT_HIGH_RISK_THRESHOLD,
        )

    if "transaction_store" not in st.session_state:
        st.session_state["transaction_store"] = TransactionStore()

    if "simulator" not in st.session_state or st.session_state["simulator"] is None:
        try:
            predictor = load_production_pipeline()
        except Exception:
            predictor = None

        try:
            st.session_state["simulator"] = TransactionSimulator(
                predictor=predictor,
                decision_engine=st.session_state["decision_engine"],
                store=st.session_state["transaction_store"],
            )
        except Exception as e:
            st.session_state["init_error"] = str(e)


def main():
    initialize_session_state()

    # Primary Navigation
    st.sidebar.markdown("### 🧭 Navigation Console")
    pages = {
        "📊 Dashboard Overview": render_dashboard,
        "📡 Live Transaction Monitoring": render_live_monitoring,
        "🚨 Fraud Alerts Queue": render_fraud_alerts,
        "🔍 Manual Transaction Scoring": render_manual_prediction,
        "📈 Model Analytics & Evaluation": render_model_analytics,
        "🗃️ Dataset Explorer & EDA": render_dataset_explorer,
    }

    selected_page_name = st.sidebar.radio(
        "Navigation",
        list(pages.keys()),
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")

    # Operational Decision Threshold Configuration
    st.sidebar.markdown("### 🎚️ Operational Risk Bands")
    engine: FraudDecisionEngine = st.session_state["decision_engine"]

    new_low = st.sidebar.slider(
        "Review Threshold (ALLOW → REVIEW)",
        min_value=0.05,
        max_value=0.50,
        value=float(engine.low_threshold),
        step=0.05,
        help="Transactions with fraud probability above this threshold trigger manual review or step-up authentication.",
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
        <div style="font-size: 12px; color: #64748b; line-height: 1.5; margin-top: 8px;">
            <span style="color: #10B981; font-weight: bold;">● ALLOW:</span> &lt; {engine.low_threshold:.0%}<br>
            <span style="color: #F59E0B; font-weight: bold;">● REVIEW:</span> {engine.low_threshold:.0%} – {engine.high_threshold:.0%}<br>
            <span style="color: #EF4444; font-weight: bold;">● BLOCK:</span> &ge; {engine.high_threshold:.0%}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    # System Status & Metadata
    store: TransactionStore = st.session_state["transaction_store"]
    stats = store.get_summary_statistics()

    st.sidebar.markdown("### ⚙️ System Status")
    st.sidebar.markdown(f"**Model:** `Random Forest v1.0`")
    st.sidebar.markdown(f"**Buffer Size:** `{stats['total_processed']} / 1,000`")
    st.sidebar.markdown(f"**Alerts Active:** `{stats['total_alerts']}`")

    # Render Active Page
    try:
        predictor = load_production_pipeline()
    except Exception as e:
        st.error(
            f"⚠️ Model pipeline error: {e}.\n\n"
            "Please ensure model artifacts are trained by running `python ml/train.py`."
        )
        return

    simulator: TransactionSimulator = st.session_state["simulator"]

    if selected_page_name == "📊 Dashboard Overview":
        render_dashboard(store, engine, simulator)
    elif selected_page_name == "📡 Live Transaction Monitoring":
        render_live_monitoring(store, engine, simulator, predictor=predictor)
    elif selected_page_name == "🚨 Fraud Alerts Queue":
        render_fraud_alerts(store, engine)
    elif selected_page_name == "🔍 Manual Transaction Scoring":
        render_manual_prediction(predictor, engine, store, simulator)
    elif selected_page_name == "📈 Model Analytics & Evaluation":
        render_model_analytics()
    elif selected_page_name == "🗃️ Dataset Explorer & EDA":
        render_dataset_explorer()


if __name__ == "__main__":
    main()
