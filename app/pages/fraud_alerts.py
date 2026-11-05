"""
Page 3: Fraud Alerts and Risk Operations Queue
Enables fraud analysts to review suspicious transactions, inspect anomaly details,
and mark transactions as Reviewed or Dismissed.
"""

import streamlit as st
import pandas as pd
from app.config import RISK_COLORS, DECISION_COLORS, REVIEW_STATUS_COLORS


def render_fraud_alerts(store, decision_engine):
    st.markdown("## 🚨 Fraud Alerts & Case Management")
    st.markdown(
        "Investigation queue for transactions flagged as **HIGH** or **MEDIUM** risk. "
        "Analysts can inspect transaction risk factors and take disposition actions."
    )

    # Filter Bar
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
    with filter_col1:
        risk_filter = st.selectbox("Risk Level", ["ALL", "HIGH", "MEDIUM"], index=0)
    with filter_col2:
        status_filter = st.selectbox("Review Status", ["ALL", "PENDING", "REVIEWED", "DISMISSED"], index=0)
    with filter_col3:
        sort_by = st.selectbox("Sort By", ["Highest Fraud Score", "Most Recent"], index=0)

    alerts = store.get_alerts(risk_filter=risk_filter, status_filter=status_filter)

    if sort_by == "Highest Fraud Score":
        alerts = sorted(alerts, key=lambda a: a["fraud_probability"], reverse=True)
    else:
        # Default order is already most recent first
        pass

    # Summary Metrics
    m1, m2, m3, m4 = st.columns(4)
    total_alerts = len(alerts)
    high_count = sum(1 for a in alerts if a["risk_level"] == "HIGH")
    pending_count = sum(1 for a in alerts if a["review_status"] == "PENDING")
    flagged_amount = sum(a["amount"] for a in alerts)

    m1.metric("Matching Alerts", total_alerts)
    m2.metric("Critical (HIGH)", high_count)
    m3.metric("Action Required (PENDING)", pending_count)
    m4.metric("Flagged Volume", f"${flagged_amount:,.2f}")

    st.markdown("---")

    if not alerts:
        st.info("No alerts found matching current filters.")
        return

    # Alert Table / Cards
    for alert in alerts:
        risk = alert["risk_level"]
        status = alert["review_status"]
        txn_id = alert["transaction_id"]
        prob = alert["fraud_probability"]
        amt = alert["amount"]
        decision = alert["decision"]

        risk_color = RISK_COLORS.get(risk, "#333")
        status_color = REVIEW_STATUS_COLORS.get(status, "#666")

        with st.expander(
            f"**[{risk}]** `{txn_id}` — Amount: **${amt:,.2f}** | Fraud Score: **{prob:.2%}** | Decision: `{decision}` | Status: `{status}`",
            expanded=(status == "PENDING" and risk == "HIGH")
        ):
            c_info, c_action = st.columns([2, 1])

            with c_info:
                st.markdown(f"**Alert ID:** `{alert['alert_id']}`")
                st.markdown(f"**Ingested Time:** `{alert['timestamp']}`")
                st.markdown(f"**Decision Engine Action:** `{decision}`")
                st.markdown(
                    f"**Risk Evaluation:** Probability `{prob:.2%}` is categorized as **{risk}** risk "
                    f"relative to thresholds (Review: `{decision_engine.low_threshold:.0%}`, Block: `{decision_engine.high_threshold:.0%}`)."
                )
                if alert.get("reviewer_notes"):
                    st.info(f"**Analyst Notes:** {alert['reviewer_notes']}")

            with c_action:
                st.markdown("##### Case Disposition")
                notes = st.text_input("Analyst Comment", key=f"note_{txn_id}", placeholder="e.g., Customer confirmed charge")
                
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("✅ Mark Reviewed", key=f"rev_{txn_id}", use_container_width=True):
                        store.update_alert_status(txn_id, "REVIEWED", notes=notes)
                        st.success("Case marked as Reviewed.")
                        st.rerun()
                with btn_col2:
                    if st.button("🚫 Dismiss (FP)", key=f"dsm_{txn_id}", use_container_width=True):
                        store.update_alert_status(txn_id, "DISMISSED", notes=notes or "False positive cleared")
                        st.warning("Case dismissed.")
                        st.rerun()

    st.markdown("---")
    if st.button("🗑️ Clear All Alerts (Reset Demo Queue)"):
        store._alerts.clear()
        st.success("Alert queue cleared.")
        st.rerun()
