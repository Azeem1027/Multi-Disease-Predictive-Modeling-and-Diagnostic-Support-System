"""
Model Comparison and Evaluation Analytics Page.
Displays cross-disease benchmarking, primary vs baseline comparison,
and diagnostic ROC/Confusion Matrix/SHAP visual artifacts.
"""
import sys
import json
from pathlib import Path

# Ensure project root and app directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    from app.ui_utils import apply_custom_styles, render_footer
except ImportError:
    from ui_utils import apply_custom_styles, render_footer
from src.config import MODELS_DIR, FIGURES_DIR

st.set_page_config(page_title="Model Benchmark Comparison", page_icon="📈", layout="wide")
apply_custom_styles()

st.title("📈 Model Comparison & Benchmark Analytics")
st.markdown("Rigorous held-out test evaluation comparing **four distinct machine learning families** across all target chronic conditions.")
st.markdown("---")

# Load all metadata JSON files
diseases = ["liver", "kidney", "diabetes", "heart"]
metadata_records = {}

for d in diseases:
    meta_path = MODELS_DIR / f"{d}_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata_records[d] = json.load(f)

if not metadata_records:
    st.warning("⚠️ No model metadata records found in `models/`. Please execute `python train_all.py` first to generate models and metrics.")
    st.stop()

# 1. Primary Model Comparison Table
st.subheader("1. Primary Model Performance Summary (Held-Out Test Set)")

summary_rows = []
for d, meta in metadata_records.items():
    m = meta.get("tuned_metrics", {})
    summary_rows.append({
        "Disease": d.capitalize(),
        "Algorithm Family": meta.get("primary_model", "N/A"),
        "Accuracy": f"{m.get('accuracy', 0.0):.4f}",
        "Recall (Sensitivity)": f"{m.get('recall', 0.0):.4f}",
        "Precision": f"{m.get('precision', 0.0):.4f}",
        "F1-Score": f"{m.get('f1', 0.0):.4f}",
        "ROC-AUC": f"{m.get('roc_auc', 0.0):.4f}",
        "Calibrated Threshold": f"{meta.get('threshold_used', 0.5):.2f}",
        "SMOTE Applied": "Yes (in CV)" if meta.get("smote_applied") else "No (balanced)"
    })

df_summary = pd.DataFrame(summary_rows)
try:
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
except Exception:
    st.table(df_summary)

# 2. Interactive Bar Chart: Primary vs Baseline
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("2. Primary Model vs Baseline Benchmark Comparison")

comparison_data = []
for d, meta in metadata_records.items():
    prim_m = meta.get("tuned_metrics", {})
    base_m = meta.get("baseline_metrics", {})
    
    # Primary
    comparison_data.append({
        "Disease": d.capitalize(),
        "Model": f"Primary: {meta.get('primary_model')}",
        "Recall": prim_m.get("recall", 0.0),
        "Precision": prim_m.get("precision", 0.0),
        "F1": prim_m.get("f1", 0.0),
        "ROC-AUC": prim_m.get("roc_auc", 0.0),
    })
    # Baseline
    comparison_data.append({
        "Disease": d.capitalize(),
        "Model": f"Baseline: {meta.get('baseline_model')}",
        "Recall": base_m.get("recall", 0.0),
        "Precision": base_m.get("precision", 0.0),
        "F1": base_m.get("f1", 0.0),
        "ROC-AUC": base_m.get("roc_auc", 0.0),
    })

df_comp = pd.DataFrame(comparison_data)

metric_to_plot = st.selectbox(
    "Select Metric for Graphical Comparison:",
    options=["Recall", "ROC-AUC", "F1", "Precision"],
    index=0
)

fig_bar = px.bar(
    df_comp,
    x="Disease",
    y=metric_to_plot,
    color="Model",
    barmode="group",
    text_auto=".3f",
    color_discrete_sequence=["#2563eb", "#9ca3af"],
    title=f"Comparative Test Performance: {metric_to_plot}"
)
fig_bar.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(range=[0, 1.05], color="#9db9bf"),
    xaxis=dict(color="#f0f7f8"),
    legend=dict(font=dict(color="#f0f7f8"))
)
st.plotly_chart(fig_bar, use_container_width=True)

# 3. Deep Dive Disease Visual Artifacts
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("3. Disease Diagnostic Artifacts (ROC, Confusion Matrix, & SHAP)")

selected_disease = st.radio("Choose Disease Cohort to Inspect:", options=["Liver", "Kidney", "Diabetes", "Heart"], horizontal=True).lower()

if selected_disease in metadata_records:
    meta = metadata_records[selected_disease]
    
    col_info1, col_info2 = st.columns([1, 1])
    with col_info1:
        st.markdown(f"#### Model Architecture: {meta.get('primary_model')}")
        st.markdown(f"* **Decision Threshold:** `{meta.get('threshold_used', 0.5):.2f}` (calibrated to prioritize recall)")
        st.markdown(f"* **SMOTE Justification:** {meta.get('smote_reason', 'N/A')}")
        if "best_params" in meta:
            st.markdown(f"* **Best Hyperparameters:** `{meta['best_params']}`")

    with col_info2:
        st.markdown("#### Test Set Performance Delta (Default vs Tuned)")
        def_m = meta.get("default_metrics", {})
        tun_m = meta.get("tuned_metrics", {})
        delta_df = pd.DataFrame([
            {"Metric": "Threshold", "Default (0.50)": "0.50", "Calibrated": f"{tun_m.get('threshold', 0.5):.2f}"},
            {"Metric": "Recall", "Default (0.50)": f"{def_m.get('recall', 0.0):.4f}", "Calibrated": f"{tun_m.get('recall', 0.0):.4f}"},
            {"Metric": "Precision", "Default (0.50)": f"{def_m.get('precision', 0.0):.4f}", "Calibrated": f"{tun_m.get('precision', 0.0):.4f}"},
            {"Metric": "F1-Score", "Default (0.50)": f"{def_m.get('f1', 0.0):.4f}", "Calibrated": f"{tun_m.get('f1', 0.0):.4f}"},
            {"Metric": "ROC-AUC", "Default (0.50)": f"{def_m.get('roc_auc', 0.0):.4f}", "Calibrated": f"{tun_m.get('roc_auc', 0.0):.4f}"},
        ])
        try:
            st.dataframe(delta_df, use_container_width=True, hide_index=True)
        except Exception:
            st.table(delta_df)

    # Figures Section
    f_col1, f_col2, f_col3 = st.columns(3)
    roc_img = FIGURES_DIR / f"{selected_disease}_roc.png"
    cm_img = FIGURES_DIR / f"{selected_disease}_cm.png"
    shap_img = FIGURES_DIR / f"{selected_disease}_shap_summary.png"

    with f_col1:
        st.markdown("**ROC Curve**")
        if roc_img.exists():
            st.image(str(roc_img), use_column_width=True)
        else:
            st.info("ROC plot will appear once trained.")

    with f_col2:
        st.markdown("**Confusion Matrix**")
        if cm_img.exists():
            st.image(str(cm_img), use_column_width=True)
        else:
            st.info("Confusion matrix will appear once trained.")

    with f_col3:
        st.markdown("**Global SHAP Summary**")
        if shap_img.exists():
            st.image(str(shap_img), use_column_width=True)
        else:
            st.info("SHAP summary will appear once trained.")

render_footer()
