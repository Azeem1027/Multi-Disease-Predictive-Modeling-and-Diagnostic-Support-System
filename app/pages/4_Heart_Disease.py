"""
Heart Disease Prediction Page - Keras Multi-Layer Perceptron (Deep Learning).
"""
import sys
from pathlib import Path

# Ensure project root and app directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
try:
    from app.ui_utils import (
        apply_custom_styles,
        render_footer,
        render_gauge_chart,
        render_shap_bar_chart,
        render_risk_badge,
        generate_explanation_text,
    )
except ImportError:
    from ui_utils import (
        apply_custom_styles,
        render_footer,
        render_gauge_chart,
        render_shap_bar_chart,
        render_risk_badge,
        generate_explanation_text,
    )
from src.inference import predict

st.set_page_config(page_title="Cardiovascular Diagnostic Support", page_icon="❤️", layout="wide")
apply_custom_styles()

st.title("❤️ Cardiovascular Heart Disease Screening")
st.markdown("Deep learning risk assessment powered by a **Multi-Layer Perceptron (Keras/TensorFlow)** trained on unified clinical cohorts (Cleveland & Hungarian).")
st.markdown("---")

col_preset1, col_preset2, _ = st.columns([1, 1, 3])
preset_mode = None
with col_preset1:
    if st.button("📋 Load Low-Risk Cardiac Profile", use_container_width=True):
        preset_mode = "low"
with col_preset2:
    if st.button("⚠️ Load Elevated Cardiac Risk Profile", use_container_width=True):
        preset_mode = "high"

defaults = {
    "age": 52, "sex": 1, "cp": 2, "trestbps": 125.0, "chol": 210.0, "fbs": 0,
    "restecg": 0, "thalach": 160.0, "exang": 0, "oldpeak": 0.5, "slope": 1,
    "ca": 0.0, "thal": 3
}

if preset_mode == "low":
    defaults.update({
        "age": 42, "sex": 0, "cp": 1, "trestbps": 115.0, "chol": 185.0, "fbs": 0,
        "restecg": 0, "thalach": 172.0, "exang": 0, "oldpeak": 0.0, "slope": 1,
        "ca": 0.0, "thal": 3
    })
elif preset_mode == "high":
    defaults.update({
        "age": 64, "sex": 1, "cp": 4, "trestbps": 155.0, "chol": 285.0, "fbs": 1,
        "restecg": 2, "thalach": 125.0, "exang": 1, "oldpeak": 2.8, "slope": 2,
        "ca": 2.0, "thal": 7
    })

with st.form("heart_form"):
    st.subheader("Cardiovascular & Hemodynamic Profile")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age (years)", min_value=20, max_value=95, value=int(defaults["age"]))
        sex = st.selectbox("Sex", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0], index=0 if defaults["sex"] == 1 else 1)[1]
        cp_options = [
            ("Typical Angina (1)", 1),
            ("Atypical Angina (2)", 2),
            ("Non-Anginal Pain (3)", 3),
            ("Asymptomatic (4)", 4)
        ]
        cp = st.selectbox("Chest Pain Type", options=cp_options, format_func=lambda x: x[0], index=int(defaults["cp"]) - 1)[1]
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80.0, max_value=230.0, value=float(defaults["trestbps"]), step=2.0)

    with c2:
        chol = st.number_input("Serum Cholesterol (mg/dL)", min_value=100.0, max_value=600.0, value=float(defaults["chol"]), step=5.0)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=[("No (<= 120 mg/dL)", 0), ("Yes (> 120 mg/dL)", 1)], format_func=lambda x: x[0], index=int(defaults["fbs"]))[1]
        restecg_options = [
            ("Normal (0)", 0),
            ("ST-T wave abnormality (1)", 1),
            ("Left Ventricular Hypertrophy (2)", 2)
        ]
        restecg = st.selectbox("Resting Electrocardiogram", options=restecg_options, format_func=lambda x: x[0], index=int(defaults["restecg"]))[1]
        thalach = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=60.0, max_value=230.0, value=float(defaults["thalach"]), step=2.0)

    with c3:
        exang = st.selectbox("Exercise Induced Angina", options=[("No", 0), ("Yes", 1)], format_func=lambda x: x[0], index=int(defaults["exang"]))[1]
        oldpeak = st.number_input("ST Depression (Exercise vs Rest)", min_value=0.0, max_value=10.0, value=float(defaults["oldpeak"]), step=0.1)
        slope_options = [("Upsloping (1)", 1), ("Flat (2)", 2), ("Downsloping (3)", 3)]
        slope = st.selectbox("Peak Exercise ST Segment Slope", options=slope_options, format_func=lambda x: x[0], index=int(defaults["slope"]) - 1)[1]
        ca = st.number_input("Major Vessels Colored by Fluoroscopy (0-3)", min_value=0.0, max_value=3.0, value=float(defaults["ca"]), step=1.0)
        thal_options = [("Normal (3)", 3), ("Fixed Defect (6)", 6), ("Reversible Defect (7)", 7)]
        thal_idx = 0 if defaults["thal"] == 3 else (1 if defaults["thal"] == 6 else 2)
        thal = st.selectbox("Thallium Stress Scintigraphy", options=thal_options, format_func=lambda x: x[0], index=thal_idx)[1]

    submitted = st.form_submit_button("⚡ Run Cardiovascular Risk Prediction", use_container_width=True, type="primary")

if submitted:
    patient_data = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": fbs,
        "restecg": restecg, "thalach": thalach, "exang": exang, "oldpeak": oldpeak,
        "slope": slope, "ca": ca, "thal": thal
    }

    with st.spinner("Executing Keras Neural Network forward pass and Kernel SHAP attribution..."):
        try:
            result = predict("heart", patient_data)
            
            st.markdown("### 📊 Diagnostic Evaluation Results")
            r_col1, r_col2 = st.columns([1, 1])

            with r_col1:
                st.markdown(f"**Clinical Risk Tier:** {render_risk_badge(result['risk_level'])}", unsafe_allow_html=True)
                fig_gauge = render_gauge_chart(result["probability"], result["threshold_used"], "Heart")
                st.plotly_chart(fig_gauge, use_container_width=True)

            with r_col2:
                fig_shap = render_shap_bar_chart(result["top_factors"])
                st.plotly_chart(fig_shap, use_container_width=True)

            st.markdown("#### 📝 Diagnostic Synthesis")
            st.info(generate_explanation_text("heart", result["probability"], result["risk_level"], result["top_factors"]))

        except Exception as e:
            st.error(f"Prediction failed: {e}. Ensure models are trained using train_all.py.")

render_footer()

