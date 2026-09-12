"""
Semiconductor Equipment Failure Intelligence - Streamlit Dashboard.

Interactive web application for semiconductor equipment engineers, process managers,
and maintenance teams. Features real-time ML failure prediction, risk scoring (LOW, MEDIUM, HIGH),
sensor telemetry trend visualizations, XAI root-cause explanations, and live scenario simulation.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.predict import FailurePredictor
from src.explainability import EquipmentExplainer


def ensure_pipeline_artifacts():
    """Generates dataset, precomputes features, and trains champion model if missing on cloud runtime."""
    feat_path = PROJECT_ROOT / "data" / "processed" / "semiconductor_equipment_features.csv"
    model_path = config.latest_model_path

    if not feat_path.exists() or not model_path.exists():
        with st.spinner("⚡ Initializing Semiconductor Telemetry Pipeline & Machine Learning Models for Streamlit Cloud..."):
            from src.data_generator import generate_and_save_dataset
            from src.preprocessing import EquipmentDataPreprocessor
            from src.features import EquipmentFeatureEngineer
            from src.train import ModelTrainerPipeline

            generate_and_save_dataset()
            preprocessor = EquipmentDataPreprocessor()
            preprocessor.process_pipeline()
            fe = EquipmentFeatureEngineer()
            fe.process_and_save()
            trainer = ModelTrainerPipeline()
            trainer.train_and_evaluate_all_models()


@st.cache_data(ttl=3600)
def load_feature_dataset() -> pd.DataFrame:
    """Loads feature dataset from disk."""
    ensure_pipeline_artifacts()
    feat_path = PROJECT_ROOT / "data" / "processed" / "semiconductor_equipment_features.csv"
    if not feat_path.exists():
        feat_path = config.processed_data_path

    if not feat_path.exists():
        st.error(f"Dataset file not found at {feat_path}. Run pipeline scripts first.")
        st.stop()

    df = pd.read_csv(feat_path)
    if config.timestamp_column in df.columns:
        df[config.timestamp_column] = pd.to_datetime(df[config.timestamp_column])
    return df


@st.cache_resource
def load_prediction_engine():
    """Initializes and caches FailurePredictor instance."""
    try:
        return FailurePredictor()
    except Exception as e:
        st.error(f"Error loading model prediction engine: {e}")
        return None


def main():
    st.set_page_config(
        page_title="Semiconductor Equipment Failure Intelligence",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom modern glassmorphic styling
    st.markdown("""
        <style>
            .stApp {
                background-color: #0f172a;
                color: #f8fafc;
            }
            .main-header {
                font-size: 2.0rem;
                font-weight: 800;
                color: #38bdf8;
                margin-bottom: 0.2rem;
            }
            .sub-header {
                font-size: 1.0rem;
                color: #94a3b8;
                margin-bottom: 1.5rem;
            }
            .metric-box {
                background: rgba(30, 41, 59, 0.8);
                border: 1px solid rgba(56, 189, 248, 0.2);
                border-radius: 10px;
                padding: 14px;
                text-align: center;
            }
            .badge-low {
                background-color: #059669;
                color: #ffffff;
                padding: 4px 12px;
                border-radius: 6px;
                font-weight: bold;
            }
            .badge-med {
                background-color: #d97706;
                color: #ffffff;
                padding: 4px 12px;
                border-radius: 6px;
                font-weight: bold;
            }
            .badge-high {
                background-color: #dc2626;
                color: #ffffff;
                padding: 4px 12px;
                border-radius: 6px;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    # Load Data & Predictor Engine
    df = load_feature_dataset()
    predictor = load_prediction_engine()
    explainer = EquipmentExplainer()

    if predictor is None:
        st.error("Prediction engine could not be initialized.")
        st.stop()

    # Header Title
    st.markdown('<div class="main-header">⚡ Semiconductor Equipment Failure Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time predictive maintenance, sensor telemetry analytics, and Explainable AI risk scoring.</div>', unsafe_allow_html=True)

    # Sidebar Equipment Selection
    st.sidebar.title("🎛️ Equipment Control")
    st.sidebar.markdown("---")

    equipment_list = sorted(df[config.id_column].unique().tolist())
    selected_eq = st.sidebar.selectbox("Select Semiconductor Equipment Unit:", equipment_list, index=0)

    # Filter data for selected equipment
    eq_df = df[df[config.id_column] == selected_eq].sort_values(config.timestamp_column).reset_index(drop=True)
    latest_reading = eq_df.iloc[-1].to_dict()

    # Run ML inference on latest reading
    explanation = explainer.explain_instance(latest_reading)
    prob_pct = explanation["failure_probability_pct"]
    prob_val = explanation["failure_probability"]
    risk_level = explanation["risk_level"]
    action_text = explanation["recommended_action"]

    # Top Status Bar Metrics
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(label="Selected Tool ID", value=selected_eq)
    with c2:
        st.metric(label="Latest Timestamp", value=str(latest_reading[config.timestamp_column])[:16])
    with c3:
        st.metric(label="Failure Risk Probability", value=prob_pct, delta="High" if risk_level == "HIGH" else "Normal")
    with c4:
        if risk_level == "LOW":
            st.markdown('### Risk Level: <span class="badge-low">LOW</span>', unsafe_allow_html=True)
        elif risk_level == "MEDIUM":
            st.markdown('### Risk Level: <span class="badge-med">MEDIUM</span>', unsafe_allow_html=True)
        else:
            st.markdown('### Risk Level: <span class="badge-high">HIGH RISK</span>', unsafe_allow_html=True)

    # Active Failure Alert Banner if Risk is HIGH or MEDIUM
    if risk_level == "HIGH":
        st.error(f"🚨 **CRITICAL EQUIPMENT ALERT ({selected_eq})**: High risk of impending failure ({prob_pct}). {action_text}")
    elif risk_level == "MEDIUM":
        st.warning(f"⚠️ **ELEVATED RISK WARNING ({selected_eq})**: Sensor drift detected ({prob_pct}). {action_text}")

    st.markdown("---")

    # KPI Telemetry Cards Grid
    st.subheader("📡 Latest Sensor Telemetry Readings")
    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)

    with k1:
        st.metric("Temp (°C)", f"{latest_reading.get('temperature', 0):.1f}", delta=f"{latest_reading.get('temp_change_1h', 0):+.1f}")
    with k2:
        st.metric("Pressure (PSI)", f"{latest_reading.get('pressure', 0):.1f}", delta=f"{latest_reading.get('press_change_1h', 0):+.1f}")
    with k3:
        st.metric("Vibration (mm/s)", f"{latest_reading.get('vibration', 0):.2f}", delta=f"{latest_reading.get('vib_change_1h', 0):+.2f}")
    with k4:
        st.metric("Voltage (V)", f"{latest_reading.get('voltage', 0):.1f}")
    with k5:
        st.metric("Current (A)", f"{latest_reading.get('current', 0):.1f}")
    with k6:
        st.metric("Speed (RPM)", f"{latest_reading.get('rpm', 0):.0f}")
    with k7:
        st.metric("Flow (L/min)", f"{latest_reading.get('flow_rate', 0):.1f}")
    with k8:
        st.metric("Runtime (hrs)", f"{latest_reading.get('runtime_hours', 0):.0f}")

    st.markdown("---")

    # Tab Views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Sensor Telemetry Trends",
        "🧠 Predictive Risk & XAI Explanation",
        "🎛️ Live Scenario Simulator",
        "📊 Fleet Status & Model Specs"
    ])

    with tab1:
        st.subheader(f"📈 Sensor Time-Series Telemetry Trends for {selected_eq}")
        st.write("Displaying historical sensor readings over time with failure threshold markers.")

        # Interactive Sub-charts
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Temperature (°C)**")
            st.line_chart(eq_df.set_index(config.timestamp_column)["temperature"], color="#e11d48")

            st.markdown("**Vibration (mm/s)**")
            st.line_chart(eq_df.set_index(config.timestamp_column)["vibration"], color="#d97706")

            st.markdown("**Current Draw (A)**")
            st.line_chart(eq_df.set_index(config.timestamp_column)["current"], color="#10b981")

        with col_b:
            st.markdown("**Pressure (PSI)**")
            st.line_chart(eq_df.set_index(config.timestamp_column)["pressure"], color="#0284c7")

            st.markdown("**Motor Speed (RPM)**")
            st.line_chart(eq_df.set_index(config.timestamp_column)["rpm"], color="#8b5cf6")

            st.markdown("**Flow Rate (L/min)**")
            st.line_chart(eq_df.set_index(config.timestamp_column)["flow_rate"], color="#ec4899")

    with tab2:
        st.subheader("🧠 Explainable AI (XAI) Risk Factors & Root Cause Analysis")
        st.write("Understand the key telemetry factors contributing to the predicted failure probability.")

        exp_col1, exp_col2 = st.columns([1, 1])

        with exp_col1:
            st.markdown(f"### Diagnostic Summary for `{selected_eq}`")
            st.code(explanation["formatted_explanation_text"], language="markdown")

        with exp_col2:
            st.markdown("### Top Risk Contributing Factors")
            factors = explanation["top_contributing_factors"]
            df_factors = pd.DataFrame(factors)[["display_name", "value", "contribution"]]
            df_factors.columns = ["Telemetry Feature", "Current Value", "Risk Impact"]
            st.dataframe(df_factors, use_container_width=True)

            # Feature Impact Bar Chart
            fig, ax = plt.subplots(figsize=(7, 3.5))
            names = [f["display_name"] for f in factors]
            contribs = [f["contribution"] for f in factors]
            ax.barh(names[::-1], contribs[::-1], color="#e11d48")
            ax.set_title("Local Risk Contributions (Log-Odds Impact)", fontweight="bold")
            ax.set_xlabel("Impact Score")
            plt.tight_layout()
            st.pyplot(fig)

    with tab3:
        st.subheader("🎛️ Real-Time Telemetry Drift Simulator")
        st.write("Manually adjust sensor parameters to evaluate real-time ML model inference and risk scoring.")

        sim_c1, sim_c2, sim_c3 = st.columns(3)

        with sim_c1:
            s_temp = st.slider("Chamber Temperature (°C)", 50.0, 140.0, float(latest_reading.get("temperature", 75.0)))
            s_press = st.slider("Chamber Pressure (PSI)", 0.0, 25.0, float(latest_reading.get("pressure", 12.0)))
            s_vib = st.slider("Vibration (mm/s)", 0.0, 8.0, float(latest_reading.get("vibration", 0.8)))

        with sim_c2:
            s_volt = st.slider("Voltage (V)", 180.0, 260.0, float(latest_reading.get("voltage", 220.0)))
            s_curr = st.slider("Current Draw (A)", 5.0, 45.0, float(latest_reading.get("current", 15.0)))
            s_rpm = st.slider("Motor Speed (RPM)", 2000.0, 4000.0, float(latest_reading.get("rpm", 3200.0)))

        with sim_c3:
            s_flow = st.slider("Flow Rate (L/min)", 10.0, 80.0, float(latest_reading.get("flow_rate", 45.0)))
            s_runtime = st.number_input("Cumulative Runtime (hrs)", 0.0, 50000.0, float(latest_reading.get("runtime_hours", 1200.0)))
            s_maint = st.number_input("Maintenance Count", 0, 50, int(latest_reading.get("maintenance_count", 3)))

        if st.button("🚀 Run Live ML Inference Simulation", use_container_width=True):
            sim_payload = {
                "equipment_id": selected_eq,
                "timestamp": str(pd.Timestamp.now())[:19],
                "temperature": s_temp,
                "pressure": s_press,
                "vibration": s_vib,
                "voltage": s_volt,
                "current": s_curr,
                "rpm": s_rpm,
                "flow_rate": s_flow,
                "runtime_hours": s_runtime,
                "maintenance_count": s_maint,
                "temp_change_1h": s_temp - latest_reading.get("temperature", 75.0),
                "press_change_1h": s_press - latest_reading.get("pressure", 12.0),
                "vib_change_1h": s_vib - latest_reading.get("vibration", 0.8),
                "current_change_1h": s_curr - latest_reading.get("current", 15.0),
                "vib_roll_mean_3h": s_vib,
                "vib_roll_mean_6h": s_vib,
                "temp_roll_mean_3h": s_temp,
                "temp_roll_mean_6h": s_temp,
                "voltage_dev": abs(s_volt - 220.0),
                "rpm_dev": 3200.0 - s_rpm,
                "flow_rate_dev": 45.0 - s_flow,
                "power_watts": s_volt * s_curr,
                "thermal_strain_index": s_temp * s_vib,
                "runtime_since_maint": s_runtime / (s_maint + 1.0),
                "maintenance_freq": s_maint / ((s_runtime / 1000.0) + 1e-5)
            }

            sim_exp = explainer.explain_instance(sim_payload)

            st.markdown("---")
            st.success("Simulation Complete!")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.metric("Simulated Failure Probability", sim_exp["failure_probability_pct"])
                st.markdown(f"**Risk Level**: `{sim_exp['risk_level']}`")
            with res_col2:
                st.code(sim_exp["formatted_explanation_text"], language="markdown")

    with tab4:
        st.subheader("📊 Fleet Overview & Model Architecture Specifications")

        st.markdown("### Fleet Risk Summary")
        fleet_records = []
        for eq_unit in equipment_list:
            sub = df[df[config.id_column] == eq_unit].iloc[-1]
            exp_u = explainer.explain_instance(sub)
            fleet_records.append({
                "Equipment ID": eq_unit,
                "Last Timestamp": str(sub[config.timestamp_column])[:16],
                "Temperature (°C)": sub["temperature"],
                "Vibration (mm/s)": sub["vibration"],
                "Failure Probability": exp_u["failure_probability_pct"],
                "Risk Category": exp_u["risk_level"]
            })

        df_fleet = pd.DataFrame(fleet_records)
        st.dataframe(df_fleet, use_container_width=True)

        st.markdown("---")
        st.markdown("### Production ML Model Specifications")
        spec_c1, spec_c2, spec_c3 = st.columns(3)
        with spec_c1:
            st.info("**Champion Model**: Logistic Regression (Balanced)")
            st.info("**Operating Threshold**: `0.25`")
        with spec_c2:
            st.info("**ROC-AUC**: `0.9871`")
            st.info("**PR-AUC**: `0.6875`")
        with spec_c3:
            st.info("**Test Set Recall**: `95.12%` (39/41 caught)")
            st.info("**Data Split**: Chronological Time-Aware (70/15/15)")


if __name__ == "__main__":
    main()
