"""
Diabetes Prediction Page - SVM with RBF Kernel (Margin-based).
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

st.set_page_config(page_title="Type-2 Diabetes Diagnostic Support", page_icon="🩸", layout="wide")
apply_custom_styles()

st.title("🩸 Type-2 Diabetes Diagnostic Screening")
st.markdown("Precision risk stratification using a **Support Vector Machine with Radial Basis Function (RBF) Kernel** calibrated on clinical glucose and metabolic markers.")
st.markdown("---")

col_preset1, col_preset2, _ = st.columns([1, 1, 3])
preset_mode = None
with col_preset1:
    if st.button("📋 Load Low-Risk Metabolic Profile", use_container_width=True):
        preset_mode = "low"
with col_preset2:
    if st.button("⚠️ Load Elevated Diabetic Risk Profile", use_container_width=True):
        preset_mode = "high"

defaults = {
    "Pregnancies": 1, "Glucose": 98.0, "BloodPressure": 72.0, "SkinThickness": 20.0,
    "Insulin": 85.0, "BMI": 24.2, "DiabetesPedigreeFunction": 0.28, "Age": 28
}

if preset_mode == "low":
    defaults.update({
        "Pregnancies": 0, "Glucose": 88.0, "BloodPressure": 68.0, "SkinThickness": 18.0,
        "Insulin": 60.0, "BMI": 22.1, "DiabetesPedigreeFunction": 0.18, "Age": 24
    })
elif preset_mode == "high":
    defaults.update({
        "Pregnancies": 5, "Glucose": 168.0, "BloodPressure": 86.0, "SkinThickness": 38.0,
        "Insulin": 240.0, "BMI": 36.5, "DiabetesPedigreeFunction": 0.85, "Age": 52
    })

with st.form("diabetes_form"):
    st.subheader("Patient Clinical & Metabolic Parameters")
    c1, c2 = st.columns(2)

    with c1:
        glucose = st.number_input("Fasting Glucose Concentration (mg/dL)", min_value=30.0, max_value=350.0, value=float(defaults["Glucose"]), step=1.0, help="2-hour oral glucose tolerance test value. Normal: 70 - 99 mg/dL")
        insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=0.0, max_value=900.0, value=float(defaults["Insulin"]), step=5.0, help="Fasting/challenge insulin level. Normal: 16 - 166 mu U/ml")
        bmi = st.number_input("Body Mass Index - BMI (kg/m²)", min_value=12.0, max_value=75.0, value=float(defaults["BMI"]), step=0.1, help="Normal: 18.5 - 24.9 kg/m²")
        bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=30.0, max_value=180.0, value=float(defaults["BloodPressure"]), step=1.0, help="Normal diastolic: < 80 mm Hg")

    with c2:
        skin = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=0.0, max_value=110.0, value=float(defaults["SkinThickness"]), step=1.0, help="Estimate of subcutaneous body fat.")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=3.0, value=float(defaults["DiabetesPedigreeFunction"]), step=0.01, help="Genetic predisposition score based on family diabetes history.")
        age = st.number_input("Age (years)", min_value=15, max_value=110, value=int(defaults["Age"]), step=1)
        preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=int(defaults["Pregnancies"]), step=1, help="Total obstetric gravidity.")

    submitted = st.form_submit_button("⚡ Evaluate Diabetic Risk", use_container_width=True, type="primary")

if submitted:
    patient_data = {
        "Pregnancies": preg,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    with st.spinner("Executing SVM RBF kernel evaluation and deriving SHAP explanations..."):
        try:
            result = predict("diabetes", patient_data)
            
            st.markdown("### 📊 Diagnostic Evaluation Results")
            r_col1, r_col2 = st.columns([1, 1])

            with r_col1:
                st.markdown(f"**Clinical Risk Tier:** {render_risk_badge(result['risk_level'])}", unsafe_allow_html=True)
                fig_gauge = render_gauge_chart(result["probability"], result["threshold_used"], "Diabetes")
                st.plotly_chart(fig_gauge, use_container_width=True)

            with r_col2:
                fig_shap = render_shap_bar_chart(result["top_factors"])
                st.plotly_chart(fig_shap, use_container_width=True)

            st.markdown("#### 📝 Diagnostic Synthesis")
            st.info(generate_explanation_text("diabetes", result["probability"], result["risk_level"], result["top_factors"]))

        except Exception as e:
            st.error(f"Prediction failed: {e}. Ensure models are trained using train_all.py.")

render_footer()

