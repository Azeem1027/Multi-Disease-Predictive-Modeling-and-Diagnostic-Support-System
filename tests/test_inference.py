"""
Unit Tests for Real-Time Inference Pipeline.
Verifies probability output in [0, 1] and valid response schema for all four diseases.
"""
import pytest
from src.inference import predict


def test_liver_inference():
    sample_patient = {
        "Age": 55,
        "Gender": "Male",
        "Total_Bilirubin": 2.1,
        "Direct_Bilirubin": 1.0,
        "Alkaline_Phosphotase": 250,
        "Alamine_Aminotransferase": 45,
        "Aspartate_Aminotransferase": 55,
        "Total_Protiens": 6.8,
        "Albumin": 3.2,
        "Albumin_and_Globulin_Ratio": 0.9,
    }
    result = predict("liver", sample_patient)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in [0, 1]
    assert result["risk_level"] in ["Low", "Moderate", "High"]
    assert isinstance(result["threshold_used"], float)
    assert isinstance(result["top_factors"], list)


def test_ckd_inference():
    sample_patient = {
        "age": 60,
        "bp": 80,
        "sg": "1.020",
        "al": "1",
        "su": "0",
        "rbc": "normal",
        "pc": "normal",
        "pcc": "notpresent",
        "ba": "notpresent",
        "bgr": 120,
        "bu": 40,
        "sc": 1.5,
        "sod": 138,
        "pot": 4.5,
        "hemo": 13.5,
        "pcv": 42,
        "wbcc": 7500,
        "rbcc": 4.8,
        "htn": "yes",
        "dm": "no",
        "cad": "no",
        "appet": "good",
        "pe": "no",
        "ane": "no",
    }
    result = predict("kidney", sample_patient)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in [0, 1]
    assert result["risk_level"] in ["Low", "Moderate", "High"]
    assert isinstance(result["top_factors"], list)


def test_diabetes_inference():
    sample_patient = {
        "Pregnancies": 2,
        "Glucose": 135,
        "BloodPressure": 75,
        "SkinThickness": 28,
        "Insulin": 120,
        "BMI": 32.5,
        "DiabetesPedigreeFunction": 0.45,
        "Age": 38,
    }
    result = predict("diabetes", sample_patient)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in [0, 1]
    assert result["risk_level"] in ["Low", "Moderate", "High"]
    assert isinstance(result["top_factors"], list)


def test_heart_inference():
    sample_patient = {
        "age": 58,
        "sex": 1,
        "cp": 3,
        "trestbps": 140,
        "chol": 240,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 0,
        "thal": 3,
    }
    result = predict("heart", sample_patient)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["label"] in [0, 1]
    assert result["risk_level"] in ["Low", "Moderate", "High"]
    assert isinstance(result["top_factors"], list)

