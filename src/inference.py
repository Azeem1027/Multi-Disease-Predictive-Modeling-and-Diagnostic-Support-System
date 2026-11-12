"""
Production Inference and Risk Prediction Module.
Provides:
predict(disease: str, patient: dict) -> {probability, label, risk_level, threshold_used, top_factors}
Loads artifacts once and caches them in memory.
Performs robust input validation and missing value handling.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

from src.config import (
    MODELS_DIR,
    LIVER_NUMERIC_FEATURES,
    LIVER_CATEGORICAL_FEATURES,
    CKD_NUMERIC_FEATURES,
    CKD_CATEGORICAL_FEATURES,
    DIABETES_NUMERIC_FEATURES,
    DIABETES_ZERO_AS_NAN,
    HEART_NUMERIC_FEATURES,
    HEART_CATEGORICAL_FEATURES,
)
from src.explain import explain_single_patient

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Global in-memory cache for models, preprocessors, and metadata
_CACHE: Dict[str, Any] = {}


def load_disease_artifacts(disease: str) -> Dict[str, Any]:
    """
    Load model artifacts, metadata, and SHAP background data once and cache them.
    """
    disease = disease.lower().strip()
    if disease in _CACHE:
        return _CACHE[disease]

    meta_file = MODELS_DIR / f"{disease}_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata file not found for {disease}: {meta_file}. Please train models first.")

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    artifacts = {"meta": meta}

    if disease == "liver":
        artifacts["pipeline"] = joblib.load(MODELS_DIR / "liver_rf_pipeline.pkl")
        bg_data = joblib.load(MODELS_DIR / "liver_shap_bg.pkl")
        artifacts["shap_bg"] = bg_data["background"]
        artifacts["feature_names"] = bg_data["feature_names"]
        artifacts["explainer_type"] = "tree"

    elif disease == "kidney":
        artifacts["pipeline"] = joblib.load(MODELS_DIR / "kidney_xgb_pipeline.pkl")
        bg_data = joblib.load(MODELS_DIR / "kidney_shap_bg.pkl")
        artifacts["shap_bg"] = bg_data["background"]
        artifacts["feature_names"] = bg_data["feature_names"]
        artifacts["explainer_type"] = "tree"

    elif disease == "diabetes":
        artifacts["pipeline"] = joblib.load(MODELS_DIR / "diabetes_svm_pipeline.pkl")
        bg_data = joblib.load(MODELS_DIR / "diabetes_shap_bg.pkl")
        artifacts["shap_bg"] = bg_data["background"]
        artifacts["feature_names"] = bg_data["feature_names"]
        artifacts["explainer_type"] = "svm"

    elif disease == "heart":
        keras_file = meta.get("keras_model_file", "heart_mlp.keras")
        keras_path = MODELS_DIR / keras_file
        artifacts["model"] = keras.models.load_model(keras_path)
        artifacts["preprocessor"] = joblib.load(MODELS_DIR / "heart_preprocessor.pkl")
        bg_data = joblib.load(MODELS_DIR / "heart_shap_bg.pkl")
        artifacts["shap_bg"] = bg_data["background"]
        artifacts["feature_names"] = bg_data["feature_names"]
        artifacts["explainer_type"] = "mlp"

    else:
        raise ValueError(f"Unknown disease: {disease}")

    _CACHE[disease] = artifacts
    logger.info("Successfully loaded and cached artifacts for %s", disease)
    return artifacts


def validate_and_prepare_input(disease: str, patient: Dict[str, Any], meta: Dict[str, Any]) -> pd.DataFrame:
    """
    Validate input types, handle clinical quirks (such as zero-as-NaN for diabetes),
    and construct a single-row DataFrame.
    """
    ranges = meta.get("feature_ranges", {})
    clean_patient = {}

    for feature, info in ranges.items():
        val = patient.get(feature, None)

        # Handle empty/missing values
        if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
            clean_patient[feature] = np.nan
            continue

        if info.get("type") == "numeric":
            try:
                num_val = float(val)
                # Diabetes zero-as-NaN check
                if disease == "diabetes" and feature in DIABETES_ZERO_AS_NAN and num_val == 0:
                    clean_patient[feature] = np.nan
                else:
                    clean_patient[feature] = num_val
            except (ValueError, TypeError):
                clean_patient[feature] = np.nan
        else:
            # Categorical string
            clean_patient[feature] = str(val).strip()

    df = pd.DataFrame([clean_patient])
    return df


def predict(disease: str, patient: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run diagnostic prediction for a patient dictionary.
    
    Returns:
    {
        "disease": str,
        "probability": float,
        "label": int (0 or 1),
        "risk_level": "Low" | "Moderate" | "High",
        "threshold_used": float,
        "top_factors": List[Dict]
    }
    """
    disease = disease.lower().strip()
    artifacts = load_disease_artifacts(disease)
    meta = artifacts["meta"]
    threshold = float(meta.get("threshold_used", 0.5))

    # Validate and build DataFrame
    patient_df = validate_and_prepare_input(disease, patient, meta)

    # Compute probability
    if disease in ["liver", "kidney", "diabetes"]:
        pipeline = artifacts["pipeline"]
        prob = float(pipeline.predict_proba(patient_df)[0, 1])
        
        # Transform for SHAP explanation
        preprocessor = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]
        transformed = preprocessor.transform(patient_df)
    else:  # heart
        model = artifacts["model"]
        preprocessor = artifacts["preprocessor"]
        transformed = preprocessor.transform(patient_df)
        prob = float(model.predict(transformed, verbose=0).flatten()[0])

    # Assign binary decision label
    label = 1 if prob >= threshold else 0

    # Clinical risk categorization
    if prob < 0.35:
        risk_level = "Low"
    elif prob < 0.65:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    # Compute top contributing factors via SHAP
    try:
        top_factors = explain_single_patient(
            model=model,
            transformed_patient=transformed,
            background_data=artifacts["shap_bg"],
            feature_names=artifacts["feature_names"],
            explainer_type=artifacts["explainer_type"],
            top_k=5
        )
    except Exception as e:
        logger.warning("Local SHAP calculation failed: %s", e)
        top_factors = []

    return {
        "disease": disease,
        "probability": round(prob, 4),
        "label": int(label),
        "risk_level": risk_level,
        "threshold_used": round(threshold, 2),
        "top_factors": top_factors,
    }

