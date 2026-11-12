"""
Global Configuration and Constants for Multi-Disease Diagnostic Support System.
"""
from pathlib import Path

# Project Directories (relative to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(r"D:\Semester 3\Artifical Intellignece\Project\Datasets")
if not DATA_DIR.exists():
    DATA_DIR = PROJECT_ROOT / "Datasets"

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ASSETS_DIR = PROJECT_ROOT / "assets"

# Fixed seed for reproducibility
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_SPLITS = 5

# ==========================================
# Disease Feature & Metadata Definitions
# ==========================================

# 1. LIVER (ILPD)
LIVER_RAW_FILE = RAW_DATA_DIR / "ilpd" / "Indian Liver Patient Dataset (ILPD).csv"
LIVER_COLUMNS = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase",
    "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio", "Dataset"
]
LIVER_NUMERIC_FEATURES = [
    "Age", "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase",
    "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio"
]
LIVER_CATEGORICAL_FEATURES = ["Gender"]
LIVER_TARGET = "Dataset"

# 2. KIDNEY (CKD)
CKD_RAW_ARFF = RAW_DATA_DIR / "ckd" / "Chronic_Kidney_Disease" / "chronic_kidney_disease.arff"
CKD_NUMERIC_FEATURES = [
    "age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"
]
CKD_CATEGORICAL_FEATURES = [
    "sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"
]
CKD_TARGET = "class"

# 3. DIABETES (PIMA INDIANS)
DIABETES_RAW_FILE = RAW_DATA_DIR / "diabetes" / "diabetes.csv"
DIABETES_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
DIABETES_NUMERIC_FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
# Features where 0 is physiologically impossible and treated as missing
DIABETES_ZERO_AS_NAN = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]
DIABETES_TARGET = "Outcome"

# 4. HEART DISEASE (CLEVELAND + HUNGARIAN)
HEART_RAW_CLEVELAND = RAW_DATA_DIR / "heart" / "processed.cleveland.data"
HEART_RAW_HUNGARIAN = RAW_DATA_DIR / "heart" / "processed.hungarian.data"
HEART_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
]
HEART_NUMERIC_FEATURES = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca"
]
HEART_CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "thal"
]
HEART_TARGET = "num"

# Common medical disclaimer
DISCLAIMER_TEXT = "Educational project. Not a medical diagnosis tool."

