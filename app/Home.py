"""
Landing Page for Multi-Disease Predictive Modeling and Diagnostic Support System.
"""
import sys
from pathlib import Path

# Ensure project root and app directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import base64
import streamlit as st
from src.config import DISCLAIMER_TEXT

st.set_page_config(
    page_title="Multi-Disease AI Diagnostic Support",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper to load and encode image for CSS background
def get_base64_image(image_path: Path) -> str:
    if image_path.exists():
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return ""

# Load CSS and Background
css_path = PROJECT_ROOT / "assets" / "style.css"
bg_path = PROJECT_ROOT / "assets" / "doctor_bg.jpg"

if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

bg_base64 = get_base64_image(bg_path)
if bg_base64:
    st.markdown(
        f"""
        <style>
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: url("data:image/jpeg;base64,{bg_base64}") center center / cover no-repeat;
            filter: brightness(0.18) saturate(0.85);
            z-index: -2;
        }}
        .stApp::after {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: linear-gradient(180deg, rgba(3, 26, 34, 0.72) 0%, rgba(3, 26, 34, 0.92) 50%, rgba(3, 26, 34, 0.98) 100%);
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Header Section
st.title("🩺 Multi-Disease Predictive Diagnostic Support System")
st.markdown(
    """
    **Early chronic disease risk estimation from routine clinical laboratory data.**  
    This system evaluates biomarker patterns across four organ systems using distinct machine learning paradigms,
    calibrated decision thresholds favoring clinical sensitivity, and local feature attribution via **SHAP**.
    """
)
st.markdown("---")

# Clickable Organ Cards
st.subheader("Select an Organ System for Clinical Screening")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        <div class="organ-card">
            <div>
                <img src="https://img.icons8.com/fluency/96/liver.png" class="organ-icon" alt="Liver" />
                <h3>Liver Disease</h3>
                <p style="color:#9db9bf; font-size:14px;">Indian Liver Patient Dataset (ILPD)</p>
                <p style="font-size:13px; color:#c4dade;">Bilirubin, SGOT, SGPT enzymes, Albumin & Globulin ratio.</p>
            </div>
            <span class="badge-low" style="margin-top:10px;">Random Forest (Bagging)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Screen Liver Risk", key="btn_liver", use_container_width=True):
        st.switch_page("pages/1_Liver_Disease.py")

with col2:
    st.markdown(
        """
        <div class="organ-card">
            <div>
                <img src="https://img.icons8.com/fluency/96/kidneys.png" class="organ-icon" alt="Kidneys" />
                <h3>Chronic Kidney</h3>
                <p style="color:#9db9bf; font-size:14px;">UCI CKD (24 Biomarkers)</p>
                <p style="font-size:13px; color:#c4dade;">Serum Creatinine, Blood Urea, Hemoglobin, Specific Gravity.</p>
            </div>
            <span class="badge-moderate" style="margin-top:10px;">XGBoost (Boosting)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Screen Kidney Risk", key="btn_kidney", use_container_width=True):
        st.switch_page("pages/2_Kidney_Disease.py")

with col3:
    st.markdown(
        """
        <div class="organ-card">
            <div>
                <img src="https://img.icons8.com/fluency/96/pancreas.png" class="organ-icon" alt="Pancreas" />
                <h3>Type-2 Diabetes</h3>
                <p style="color:#9db9bf; font-size:14px;">Pima Indians Dataset</p>
                <p style="font-size:13px; color:#c4dade;">Fasting Glucose, Serum Insulin, BMI, Pedigree, Age.</p>
            </div>
            <span class="badge-low" style="margin-top:10px;">SVM RBF (Margin-based)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Screen Diabetes Risk", key="btn_diabetes", use_container_width=True):
        st.switch_page("pages/3_Diabetes.py")

with col4:
    st.markdown(
        """
        <div class="organ-card">
            <div>
                <img src="https://img.icons8.com/fluency/96/heart-with-pulse.png" class="organ-icon" alt="Heart" />
                <h3>Heart Disease</h3>
                <p style="color:#9db9bf; font-size:14px;">Cleveland + Hungarian Cohorts</p>
                <p style="font-size:13px; color:#c4dade;">Resting BP, Cholesterol, Thalassemia, ST Depression.</p>
            </div>
            <span class="badge-high" style="margin-top:10px;">Keras MLP (Deep Learning)</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Screen Heart Risk", key="btn_heart", use_container_width=True):
        st.switch_page("pages/4_Heart_Disease.py")

st.markdown("<br>", unsafe_allow_html=True)

# System Architecture & Key Engineering Highlights
with st.expander("🔬 System Architecture & Clinical Methodology", expanded=True):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🛡️ Leakage-Free Preprocessing")
        st.markdown(
            """
            * **Strict Stratified Split:** 80/20 train/test split executed prior to any scaling or imputation.
            * **Fold-Isolated SMOTE:** Imputation, scaling, and oversampling occur strictly inside cross-validation folds using `imblearn.pipeline.Pipeline`.
            * **Quirk Remediation:** Impossible zeros in diabetes labs converted to missing; trailing commas and tabs sanitized in CKD.
            """
        )
    with col_b:
        st.markdown("#### 🎯 Clinical Calibration & Explainability")
        st.markdown(
            """
            * **Threshold Tuning:** Decision thresholds tuned per disease to maximize sensitivity ($F_2$-score) to avoid false negatives in critical screenings.
            * **SHAP Explainability:** TreeExplainer and KernelExplainer pinpoint top contributing biomarkers with exact direction and magnitude.
            * **Model Benchmarking:** Every primary model is compared against a Logistic Regression / LightGBM baseline on the held-out test set.
            """
        )

# Navigation Quick Links
st.info("💡 Tip: Navigate to **Model Comparison** in the sidebar to review test metrics, ROC curves, and confusion matrices across all four disease models.")

# Medical Footer
st.markdown(f'<div class="med-footer">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)

