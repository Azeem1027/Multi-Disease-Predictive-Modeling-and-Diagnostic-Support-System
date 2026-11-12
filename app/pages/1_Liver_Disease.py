"""
Liver Disease Prediction Page - Random Forest with Bagging.
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

st.set_page_config(page_title="Liver Disease Diagnostic Support", page_icon="🧪", layout="wide")
apply_custom_styles()

st.title("🧪 Liver Disease Diagnostic Screening")
st.markdown("Predictive modeling of hepatic dysfunction using an **ensemble Random Forest classifier** on routine liver function tests (LFT).")
st.markdown("---")

# Sample Presets
col_preset1, col_preset2, _ = st.columns([1, 1, 3])
preset_mode = None
with col_preset1:
    if st.button("📋 Load Typical Low-Risk Case", use_container_width=True):
        preset_mode = "low"
with col_preset2:
    if st.button("⚠️ Load Elevated-Risk Case", use_container_width=True):
        preset_mode = "high"

# Default baseline values
defaults = {
    "Age": 45, "Gender": "Male", "Total_Bilirubin": 0.8, "Direct_Bilirubin": 0.2,
    "Alkaline_Phosphotase": 180, "Alamine_Aminotransferase": 25, "Aspartate_Aminotransferase": 28,
    "Total_Protiens": 7.0, "Albumin": 3.8, "Albumin_and_Globulin_Ratio": 1.1
}

if preset_mode == "low":
    defaults.update({
        "Age": 35, "Gender": "Female", "Total_Bilirubin": 0.6, "Direct_Bilirubin": 0.1,
        "Alkaline_Phosphotase": 160, "Alamine_Aminotransferase": 20, "Aspartate_Aminotransferase": 22,
        "Total_Protiens": 7.2, "Albumin": 4.1, "Albumin_and_Globulin_Ratio": 1.3
    })
elif preset_mode == "high":
    defaults.update({
        "Age": 62, "Gender": "Male", "Total_Bilirubin": 4.5, "Direct_Bilirubin": 2.2,
        "Alkaline_Phosphotase": 480, "Alamine_Aminotransferase": 95, "Aspartate_Aminotransferase": 110,
        "Total_Protiens": 6.1, "Albumin": 2.7, "Albumin_and_Globulin_Ratio": 0.65
    })

# Form Layout
with st.form("liver_form"):
    st.subheader("Patient Biomarkers & Clinical Chemistry")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age (years)", min_value=1, max_value=105, value=int(defaults["Age"]), help="Patient age in completed years.")
        gender = st.selectbox("Gender", options=["Male", "Female"], index=0 if defaults["Gender"] == "Male" else 1)
        tot_bil = st.number_input("Total Bilirubin (mg/dL)", min_value=0.1, max_value=75.0, value=float(defaults["Total_Bilirubin"]), step=0.1, help="Normal range: 0.2 - 1.2 mg/dL")
        dir_bil = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, max_value=25.0, value=float(defaults["Direct_Bilirubin"]), step=0.1, help="Conjugated bilirubin. Normal range: 0.0 - 0.3 mg/dL")

    with c2:
        alp = st.number_input("Alkaline Phosphatase - ALP (IU/L)", min_value=10, max_value=2500, value=int(defaults["Alkaline_Phosphotase"]), step=5, help="Enzyme related to bile ducts. Normal: 44 - 147 IU/L")
        sgpt = st.number_input("Alamine Aminotransferase - SGPT (IU/L)", min_value=5, max_value=2000, value=int(defaults["Alamine_Aminotransferase"]), step=5, help="ALT enzyme. Normal: 7 - 56 IU/L")
        sgot = st.number_input("Aspartate Aminotransferase - SGOT (IU/L)", min_value=5, max_value=4000, value=int(defaults["Aspartate_Aminotransferase"]), step=5, help="AST enzyme. Normal: 10 - 40 IU/L")

    with c3:
        tp = st.number_input("Total Proteins (g/dL)", min_value=2.0, max_value=12.0, value=float(defaults["Total_Protiens"]), step=0.1, help="Normal range: 6.0 - 8.3 g/dL")
        alb = st.number_input("Albumin (g/dL)", min_value=0.5, max_value=7.0, value=float(defaults["Albumin"]), step=0.1, help="Synthesized by liver. Normal: 3.4 - 5.4 g/dL")
        agr = st.number_input("Albumin and Globulin Ratio", min_value=0.1, max_value=4.0, value=float(defaults["Albumin_and_Globulin_Ratio"]), step=0.05, help="Normal: 1.0 - 2.2")

    submitted = st.form_submit_button("⚡ Run Diagnostic Risk Evaluation", use_container_width=True, type="primary")

if submitted:
    patient_data = {
        "Age": age,
        "Gender": gender,
        "Total_Bilirubin": tot_bil,
        "Direct_Bilirubin": dir_bil,
        "Alkaline_Phosphotase": alp,
        "Alamine_Aminotransferase": sgpt,
        "Aspartate_Aminotransferase": sgot,
        "Total_Protiens": tp,
        "Albumin": alb,
        "Albumin_and_Globulin_Ratio": agr,
    }

    with st.spinner("Analyzing hepatic biomarker profile and calculating SHAP contributions..."):
        try:
            result = predict("liver", patient_data)
            
            st.markdown("### 📊 Diagnostic Evaluation Results")
            r_col1, r_col2 = st.columns([1, 1])

            with r_col1:
                st.markdown(f"**Clinical Risk Tier:** {render_risk_badge(result['risk_level'])}", unsafe_allow_html=True)
                fig_gauge = render_gauge_chart(result["probability"], result["threshold_used"], "Liver")
                st.plotly_chart(fig_gauge, use_container_width=True)

            with r_col2:
                fig_shap = render_shap_bar_chart(result["top_factors"])
                st.plotly_chart(fig_shap, use_container_width=True)

            # Plain-Language Clinical Synthesis
            st.markdown("#### 📝 Diagnostic Synthesis")
            st.info(generate_explanation_text("liver", result["probability"], result["risk_level"], result["top_factors"]))

        except Exception as e:
            st.error(f"Prediction failed: {e}. Ensure models are trained using train_all.py.")

render_footer()

