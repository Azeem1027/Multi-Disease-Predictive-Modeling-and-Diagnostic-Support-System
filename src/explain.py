"""
SHAP Explainability Module for All Four Diseases.
Provides:
1. Global summary plot generation (TreeExplainer for RF/XGBoost, KernelExplainer for SVM/MLP).
2. Per-patient local explanation (top contributing factors with magnitude and direction).
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from src.config import FIGURES_DIR

logger = logging.getLogger(__name__)


def generate_global_shap_summary(
    model: Any,
    X_train_trans: np.ndarray,
    X_test_trans: np.ndarray,
    feature_names: List[str],
    disease_name: str,
    explainer_type: str = "tree",
    max_display: int = 10,
    background_samples: int = 50,
    test_eval_samples: int = 60
) -> Path:
    """
    Generate and save global SHAP summary plot.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_file = FIGURES_DIR / f"{disease_name.lower()}_shap_summary.png"
    logger.info("Generating global SHAP summary for %s using %s explainer...", disease_name, explainer_type)

    plt.figure()
    
    if explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
        # Use up to 100 samples for swift calculation
        eval_data = X_test_trans[:min(100, len(X_test_trans))]
        shap_values = explainer.shap_values(eval_data)
        
        # Binary classification handling: get class 1 (disease) SHAP values
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_matrix = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_matrix = shap_values[:, :, 1]
        else:
            shap_matrix = shap_values
            
        shap.summary_plot(shap_matrix, eval_data, feature_names=feature_names,
                          max_display=max_display, show=False)
    
    elif explainer_type in ["kernel", "svm", "mlp"]:
        # Subsample background for computational speed on laptop
        bg_idx = np.random.choice(len(X_train_trans), min(background_samples, len(X_train_trans)), replace=False)
        background = X_train_trans[bg_idx]
        
        # Prediction wrapper
        if hasattr(model, "predict_proba"):
            def predict_fn(x):
                return model.predict_proba(x)[:, 1]
        elif hasattr(model, "predict"):
            def predict_fn(x):
                preds = model.predict(x)
                return preds.flatten()
        else:
            predict_fn = model

        explainer = shap.KernelExplainer(predict_fn, background)
        eval_idx = np.random.choice(len(X_test_trans), min(test_eval_samples, len(X_test_trans)), replace=False)
        eval_data = X_test_trans[eval_idx]
        
        shap_values = explainer.shap_values(eval_data, nsamples=100)
        shap.summary_plot(shap_values, eval_data, feature_names=feature_names,
                          max_display=max_display, show=False)
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")

    plt.title(f"{disease_name.capitalize()} - SHAP Biomarker Contributions", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved SHAP summary to %s", out_file)
    return out_file


def explain_single_patient(
    model: Any,
    transformed_patient: np.ndarray,
    background_data: np.ndarray,
    feature_names: List[str],
    explainer_type: str = "tree",
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Compute local SHAP explanation for a single patient.
    Returns sorted list of top contributing biomarkers with direction and magnitude.
    """
    if explainer_type == "tree":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_patient)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            vals = shap_values[1][0]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 2:
            vals = shap_values[0]
        else:
            vals = np.array(shap_values).flatten()
    else:
        if hasattr(model, "predict_proba"):
            predict_fn = lambda x: model.predict_proba(x)[:, 1]
        elif hasattr(model, "predict"):
            predict_fn = lambda x: model.predict(x).flatten()
        else:
            predict_fn = model
            
        bg = background_data[:min(30, len(background_data))]
        explainer = shap.KernelExplainer(predict_fn, bg)
        shap_values = explainer.shap_values(transformed_patient, nsamples=80)
        vals = np.array(shap_values).flatten()

    factors = []
    for name, impact, val in zip(feature_names, vals, transformed_patient[0]):
        direction = "raises risk" if impact > 0 else "lowers risk"
        factors.append({
            "feature": name,
            "impact": float(impact),
            "abs_impact": float(abs(impact)),
            "direction": direction,
            "normalized_value": float(val)
        })

    # Sort descending by absolute contribution magnitude
    factors.sort(key=lambda x: x["abs_impact"], reverse=True)
    return factors[:top_k]

