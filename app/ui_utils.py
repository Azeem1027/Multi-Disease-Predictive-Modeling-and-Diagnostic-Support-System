"""
Reusable UI Components and Visualizations for Streamlit Disease Pages.
"""
from pathlib import Path
from typing import Dict, Any, List
import streamlit as st
import plotly.graph_objects as go
from src.config import PROJECT_ROOT, DISCLAIMER_TEXT


def apply_custom_styles():
    """Load and inject shared CSS styles."""
    css_path = PROJECT_ROOT / "assets" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_footer():
    """Render mandatory educational disclaimer footer."""
    st.markdown(f'<div class="med-footer">{DISCLAIMER_TEXT}</div>', unsafe_allow_html=True)


def render_gauge_chart(probability: float, threshold: float, disease_name: str) -> go.Figure:
    """
    Render Plotly Gauge chart displaying disease risk percentage,
    risk zones, and the calibrated decision threshold marker.
    """
    pct = round(probability * 100, 1)
    thresh_pct = round(threshold * 100, 1)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 42, "color": "#f0f7f8"}},
        delta={"reference": thresh_pct, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
        title={"text": f"<b>{disease_name.capitalize()} Risk Score</b><br><span style='font-size:12px;color:#9db9bf'>Calibrated Decision Threshold: {thresh_pct}%</span>", "font": {"size": 16, "color": "#8fe3ea"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9db9bf"},
            "bar": {"color": "#8fe3ea", "thickness": 0.28},
            "bgcolor": "rgba(7, 40, 51, 0.6)",
            "borderwidth": 1,
            "bordercolor": "rgba(143, 227, 234, 0.3)",
            "steps": [
                {"range": [0, 35], "color": "rgba(16, 185, 129, 0.25)"},
                {"range": [35, 65], "color": "rgba(245, 158, 11, 0.25)"},
                {"range": [65, 100], "color": "rgba(239, 68, 68, 0.25)"},
            ],
            "threshold": {
                "line": {"color": "#f43f5e", "width": 4},
                "thickness": 0.75,
                "value": thresh_pct
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=50, b=20),
        height=260,
    )
    return fig


def render_shap_bar_chart(factors: List[Dict[str, Any]]) -> go.Figure:
    """
    Render Plotly horizontal bar chart of top SHAP feature contributions.
    """
    if not factors:
        fig = go.Figure()
        fig.add_annotation(text="No SHAP factor data available", showarrow=False, font=dict(color="#9db9bf"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        return fig

    # Reverse list so the highest impact feature appears at the top
    factors_rev = list(reversed(factors))
    feature_names = [f["feature"] for f in factors_rev]
    impacts = [f["impact"] for f in factors_rev]
    colors = ["#ef4444" if imp > 0 else "#10b981" for imp in impacts]

    fig = go.Figure(go.Bar(
        x=impacts,
        y=feature_names,
        orientation="h",
        marker_color=colors,
        text=[f"{imp:+.3f}" for imp in impacts],
        textposition="auto",
    ))

    fig.update_layout(
        title={"text": "<b>Key Biomarker Contributions (SHAP Values)</b>", "font": {"size": 14, "color": "#8fe3ea"}},
        xaxis=dict(
            title="SHAP Impact (Pushes Risk Higher ➡️ / Lowers Risk ⬅️)",
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.4)",
            color="#9db9bf"
        ),
        yaxis=dict(color="#f0f7f8"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=20, t=40, b=20),
        height=260,
    )
    return fig


def render_risk_badge(risk_level: str) -> str:
    """Return HTML string for styled risk level badge."""
    level = risk_level.lower()
    if level == "low":
        return '<span class="badge-low">🟢 Low Estimated Risk</span>'
    elif level == "moderate":
        return '<span class="badge-moderate">🟡 Moderate Estimated Risk</span>'
    else:
        return '<span class="badge-high">🔴 High Estimated Risk</span>'


def generate_explanation_text(disease: str, prob: float, risk_level: str, factors: List[Dict[str, Any]]) -> str:
    """
    Generate plain-language clinical summary explaining the prediction.
    """
    pct = round(prob * 100, 1)
    raising = [f["feature"] for f in factors if f["impact"] > 0]
    lowering = [f["feature"] for f in factors if f["impact"] < 0]

    raising_str = ", ".join(f"`{feat}`" for feat in raising[:3]) if raising else "none"
    lowering_str = ", ".join(f"`{feat}`" for feat in lowering[:3]) if lowering else "none"

    if risk_level == "High":
        text = (
            f"Based on the provided laboratory measurements, the model indicates an **elevated probability ({pct}%)** "
            f"of **{disease.capitalize()} Disease**. The primary biomarkers driving this elevated risk score are "
            f"{raising_str}. Comprehensive clinical follow-up and diagnostic confirmation are strongly advised."
        )
    elif risk_level == "Moderate":
        text = (
            f"The assessment calculated a **moderate risk score ({pct}%)** for **{disease.capitalize()} Disease**. "
            f"Biomarkers pushing risk upwards include {raising_str}, partially balanced by protective indicators {lowering_str}. "
            f"Periodic lab monitoring and lifestyle review are recommended."
        )
    else:
        text = (
            f"The evaluation shows a **favorable low-risk probability ({pct}%)** for **{disease.capitalize()} Disease**. "
            f"Key stabilizing biomarkers include {lowering_str}. Continue routine health screenings."
        )
    return text

