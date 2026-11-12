"""
Kidney Disease Prediction Page - XGBoost with Boosting.
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

st.set_page_config(page_title="Chronic Kidney Disease Support", page_icon="🫘", layout="wide")
apply_custom_styles()

st.title("🫘 Chronic Kidney Disease (CKD) Screening")
st.markdown("Advanced multi-biomarker renal evaluation using gradient boosted decision trees (**XGBoost** with LightGBM benchmark).")
st.markdown("---")

# Sample Presets
col_preset1, col_preset2, _ = st.columns([1, 1, 3])
preset_mode = None
with col_preset1:
    if st.button("📋 Load Low-Risk Renal Profile", use_container_width=True):
        preset_mode = "low"
with col_preset2:
    if st.button("⚠️ Load Elevated CKD Risk Profile", use_container_width=True):
        preset_mode = "high"

defaults = {
    "age": 48, "bp": 80, "bgr": 110, "bu": 32, "sc": 0.9, "sod": 140, "pot": 4.2,
    "hemo": 15.0, "pcv": 45, "wbcc": 7200, "rbcc": 5.2,
    "sg": "1.020", "al": "0", "su": "0", "rbc": "normal", "pc": "normal",
    "pcc": "notpresent", "ba": "notpresent", "htn": "no", "dm": "no",
    "cad": "no", "appet": "good", "pe": "no", "ane": "no"
}

if preset_mode == "low":
    defaults.update({
        "age": 36, "bp": 70, "bgr": 95, "bu": 24, "sc": 0.8, "sod": 142, "pot": 4.1,
        "hemo": 15.8, "pcv": 48, "wbcc": 6800, "rbcc": 5.4,
        "sg": "1.025", "al": "0", "su": "0", "rbc": "normal", "pc": "normal",
        "pcc": "notpresent", "ba": "notpresent", "htn": "no", "dm": "no",
        "cad": "no", "appet": "good", "pe": "no", "ane": "no"
    })
elif preset_mode == "high":
    defaults.update({
        "age": 65, "bp": 90, "bgr": 210, "bu": 75, "sc": 3.8, "sod": 131, "pot": 5.8,
        "hemo": 9.4, "pcv": 29, "wbcc": 11200, "rbcc": 3.4,
        "sg": "1.010", "al": "3", "su": "2", "rbc": "abnormal", "pc": "abnormal",
        "pcc": "present", "ba": "present", "htn": "yes", "dm": "yes",
        "cad": "yes", "appet": "poor", "pe": "yes", "ane": "yes"
    })

with st.form("ckd_form"):
    tab1, tab2, tab3 = st.tabs(["🧪 Blood Chemistry & Renal Function", "🔬 Urinalysis Panel", "📋 Clinical History & Physical"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.2, max_value=25.0, value=float(defaults["sc"]), step=0.1, help="Critical renal filtration biomarker. Normal: 0.6 - 1.2 mg/dL")
            bu = st.number_input("Blood Urea (mg/dL)", min_value=1.0, max_value=300.0, value=float(defaults["bu"]), step=1.0, help="Urea nitrogen. Normal: 7 - 20 mg/dL")
            bgr = st.number_input("Blood Glucose Random (mg/dL)", min_value=20.0, max_value=500.0, value=float(defaults["bgr"]), step=5.0)
        with c2:
            sod = st.number_input("Sodium (mEq/L)", min_value=90.0, max_value=180.0, value=float(defaults["sod"]), step=1.0, help="Normal: 135 - 145 mEq/L")
            pot = st.number_input("Potassium (mEq/L)", min_value=1.5, max_value=15.0, value=float(defaults["pot"]), step=0.1, help="Normal: 3.5 - 5.0 mEq/L")
            hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=float(defaults["hemo"]), step=0.1, help="Normal: 12 - 17 g/dL")
        with c3:
            pcv = st.number_input("Packed Cell Volume - PCV (%)", min_value=10.0, max_value=65.0, value=float(defaults["pcv"]), step=1.0)
            wbcc = st.number_input("White Blood Cell Count (cells/cumm)", min_value=1000.0, max_value=30000.0, value=float(defaults["wbcc"]), step=200.0)
            rbcc = st.number_input("Red Blood Cell Count (millions/cmm)", min_value=1.0, max_value=8.5, value=float(defaults["rbcc"]), step=0.1)

    with tab2:
        u1, u2 = st.columns(2)
        with u1:
            sg = st.selectbox("Specific Gravity", options=["1.005", "1.010", "1.015", "1.020", "1.025"], index=["1.005", "1.010", "1.015", "1.020", "1.025"].index(defaults["sg"]))
            al = st.selectbox("Albumin (Proteinuria Score)", options=["0", "1", "2", "3", "4", "5"], index=int(defaults["al"]))
            su = st.selectbox("Sugar (Glucosuria Score)", options=["0", "1", "2", "3", "4", "5"], index=int(defaults["su"]))
        with u2:
            rbc = st.selectbox("Red Blood Cells", options=["normal", "abnormal"], index=0 if defaults["rbc"] == "normal" else 1)
            pc = st.selectbox("Pus Cells", options=["normal", "abnormal"], index=0 if defaults["pc"] == "normal" else 1)
            pcc = st.selectbox("Pus Cell Clumps", options=["notpresent", "present"], index=0 if defaults["pcc"] == "notpresent" else 1)
            ba = st.selectbox("Bacteria", options=["notpresent", "present"], index=0 if defaults["ba"] == "notpresent" else 1)

    with tab3:
        h1, h2 = st.columns(2)
        with h1:
            age = st.number_input("Age (years)", min_value=1, max_value=100, value=int(defaults["age"]))
            bp = st.number_input("Blood Pressure (mm/Hg)", min_value=40, max_value=220, value=int(defaults["bp"]), step=5)
            htn = st.selectbox("Hypertension", options=["no", "yes"], index=0 if defaults["htn"] == "no" else 1)
            dm = st.selectbox("Diabetes Mellitus", options=["no", "yes"], index=0 if defaults["dm"] == "no" else 1)
        with h2:
            cad = st.selectbox("Coronary Artery Disease", options=["no", "yes"], index=0 if defaults["cad"] == "no" else 1)
            appet = st.selectbox("Appetite", options=["good", "poor"], index=0 if defaults["appet"] == "good" else 1)
            pe = st.selectbox("Pedal Edema", options=["no", "yes"], index=0 if defaults["pe"] == "no" else 1)
            ane = st.selectbox("Anemia", options=["no", "yes"], index=0 if defaults["ane"] == "no" else 1)

    submitted = st.form_submit_button("⚡ Evaluate Renal Risk", use_container_width=True, type="primary")

if submitted:
    patient_data = {
        "age": age, "bp": bp, "bgr": bgr, "bu": bu, "sc": sc, "sod": sod, "pot": pot,
        "hemo": hemo, "pcv": pcv, "wbcc": wbcc, "rbcc": rbcc,
        "sg": sg, "al": al, "su": su, "rbc": rbc, "pc": pc, "pcc": pcc, "ba": ba,
        "htn": htn, "dm": dm, "cad": cad, "appet": appet, "pe": pe, "ane": ane
    }

    with st.spinner("Executing XGBoost inference and computing local feature SHAP values..."):
        try:
            result = predict("kidney", patient_data)
            
            st.markdown("### 📊 Diagnostic Evaluation Results")
            r_col1, r_col2 = st.columns([1, 1])

            with r_col1:
                st.markdown(f"**Clinical Risk Tier:** {render_risk_badge(result['risk_level'])}", unsafe_allow_html=True)
                fig_gauge = render_gauge_chart(result["probability"], result["threshold_used"], "Kidney")
                st.plotly_chart(fig_gauge, use_container_width=True)

            with r_col2:
                fig_shap = render_shap_bar_chart(result["top_factors"])
                st.plotly_chart(fig_shap, use_container_width=True)

            st.markdown("#### 📝 Diagnostic Synthesis")
            st.info(generate_explanation_text("kidney", result["probability"], result["risk_level"], result["top_factors"]))

        except Exception as e:
            st.error(f"Prediction failed: {e}. Ensure models are trained using train_all.py.")

render_footer()

