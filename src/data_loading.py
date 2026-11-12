"""
Robust Data Loading and Ingestion Module for All Four Diseases.
Ensures zero data leakage and preserves raw data integrity.
"""
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

from src.config import (
    LIVER_RAW_FILE,
    LIVER_COLUMNS,
    CKD_RAW_ARFF,
    CKD_NUMERIC_FEATURES,
    CKD_CATEGORICAL_FEATURES,
    DIABETES_RAW_FILE,
    DIABETES_COLUMNS,
    DIABETES_ZERO_AS_NAN,
    HEART_RAW_CLEVELAND,
    HEART_RAW_HUNGARIAN,
    HEART_COLUMNS,
    PROCESSED_DATA_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_liver_data(save_processed: bool = True) -> pd.DataFrame:
    """
    Load and parse Indian Liver Patient Dataset (ILPD).
    
    Quirks handled:
    - Gender encoding (Female/Male strings)
    - Missing values in Albumin_and_Globulin_Ratio (empty or '?')
    - Target mapping: 1 (patient) -> 1, 2 (non-patient) -> 0
    """
    logger.info("Loading ILPD Liver dataset from %s", LIVER_RAW_FILE)
    if not LIVER_RAW_FILE.exists():
        raise FileNotFoundError(f"ILPD file not found: {LIVER_RAW_FILE}")

    # ILPD raw csv does not have headers
    df = pd.read_csv(LIVER_RAW_FILE, header=None, names=LIVER_COLUMNS)
    
    # Clean whitespace in strings
    df["Gender"] = df["Gender"].astype(str).str.strip()
    
    # Map target: 1 = patient -> 1, 2 = non-patient -> 0
    df["Dataset"] = df["Dataset"].map({1: 1, 2: 0})
    if df["Dataset"].isnull().any():
        raise ValueError("Invalid target values found in ILPD Dataset column")

    # Ensure numeric columns are properly typed
    numeric_cols = [c for c in LIVER_COLUMNS if c not in ["Gender", "Dataset"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Liver dataset loaded successfully. Shape: %s. Patient cases: %d, Non-patient: %d",
                df.shape, (df["Dataset"] == 1).sum(), (df["Dataset"] == 0).sum())

    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_DIR / "liver_cleaned.csv", index=False)

    return df


def load_ckd_data(save_processed: bool = True) -> pd.DataFrame:
    """
    Load and parse Chronic Kidney Disease (CKD) ARFF dataset.
    
    Quirks handled:
    - '?' and stray tabs ('\\t') or whitespace
    - Trailing commas in ARFF data lines (rows 69 and 72)
    - Mixed datatypes (e.g. pcv, wbcc, rbcc encoded as strings with tabs)
    - Target variable 'class' with values 'ckd'/'notckd' (stripped of whitespace/tabs) -> 1 / 0
    """
    logger.info("Loading CKD dataset from %s", CKD_RAW_ARFF)
    if not CKD_RAW_ARFF.exists():
        raise FileNotFoundError(f"CKD ARFF file not found: {CKD_RAW_ARFF}")

    attributes = []
    data_rows = []
    is_data = False

    with open(CKD_RAW_ARFF, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            clean_line = line.strip()
            if not clean_line:
                continue
            if clean_line.lower().startswith("@attribute"):
                parts = clean_line.split()
                attr_name = parts[1].strip("'\"")
                attributes.append(attr_name)
            elif clean_line.lower().startswith("@data"):
                is_data = True
            elif is_data:
                # Handle stray spaces/tabs
                clean_data_line = clean_line.replace("\t", "").strip()
                # Handle accidental double commas (e.g. 'no,,no')
                while ",," in clean_data_line:
                    clean_data_line = clean_data_line.replace(",,", ",")
                raw_vals = [v.strip().strip("'\"") for v in clean_data_line.split(",")]
                # Filter out trailing empty string if line ended with comma
                if len(raw_vals) == len(attributes) + 1 and raw_vals[-1] == "":
                    raw_vals = raw_vals[:-1]
                if len(raw_vals) == len(attributes):
                    data_rows.append(raw_vals)
                else:
                    logger.warning("Skipping malformed row with %d fields (expected %d): %s",
                                   len(raw_vals), len(attributes), clean_line)

    df = pd.DataFrame(data_rows, columns=attributes)

    # Clean target
    df["class"] = df["class"].astype(str).str.strip().str.lower()
    df["class"] = df["class"].replace({"ckd": 1, "notckd": 0})
    # Ensure binary integer target
    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    df = df.dropna(subset=["class"])
    df["class"] = df["class"].astype(int)

    # Clean numeric features
    for col in CKD_NUMERIC_FEATURES:
        if col in df.columns:
            # Replace '?' with NaN and cast to float
            df[col] = df[col].replace("?", np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean categorical features
    for col in CKD_CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].replace("?", np.nan)
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({"nan": np.nan, "": np.nan})

    logger.info("CKD dataset loaded successfully. Shape: %s. CKD cases: %d, Non-CKD: %d",
                df.shape, (df["class"] == 1).sum(), (df["class"] == 0).sum())

    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_DIR / "ckd_cleaned.csv", index=False)

    return df


def load_diabetes_data(save_processed: bool = True) -> pd.DataFrame:
    """
    Load Pima Indians Diabetes Dataset.
    
    Quirks handled:
    - Impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI
      are treated as physiologically missing (NaN) for median imputation.
    """
    logger.info("Loading Diabetes dataset from %s", DIABETES_RAW_FILE)
    if not DIABETES_RAW_FILE.exists():
        raise FileNotFoundError(f"Diabetes file not found: {DIABETES_RAW_FILE}")

    # Check if header exists
    first_line = open(DIABETES_RAW_FILE, "r").readline().strip()
    if "pregnancies" in first_line.lower() or "glucose" in first_line.lower():
        df = pd.read_csv(DIABETES_RAW_FILE)
    else:
        df = pd.read_csv(DIABETES_RAW_FILE, header=None, names=DIABETES_COLUMNS)

    # Standardize column names
    df.columns = DIABETES_COLUMNS

    # Cast all to float/int
    for col in DIABETES_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Treat physiologically impossible zeros as NaN
    for col in DIABETES_ZERO_AS_NAN:
        zero_count = (df[col] == 0).sum()
        logger.info("Converting %d impossible zeros in %s to NaN for imputation", zero_count, col)
        df[col] = df[col].replace(0, np.nan)

    df["Outcome"] = df["Outcome"].astype(int)

    logger.info("Diabetes dataset loaded successfully. Shape: %s. Diabetic cases: %d, Non-diabetic: %d",
                df.shape, (df["Outcome"] == 1).sum(), (df["Outcome"] == 0).sum())

    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_DIR / "diabetes_cleaned.csv", index=False)

    return df


def load_heart_data(save_processed: bool = True) -> pd.DataFrame:
    """
    Load and combine Cleveland and Hungarian Heart Disease datasets.
    
    Quirks handled:
    - Missing values represented as '?'
    - Binarize target 'num': > 0 -> 1 (disease), == 0 -> 0 (no disease)
    - Multi-cohort unification
    """
    logger.info("Loading Heart Disease datasets from Cleveland & Hungarian sources")
    if not HEART_RAW_CLEVELAND.exists() or not HEART_RAW_HUNGARIAN.exists():
        raise FileNotFoundError("Heart disease raw files missing")

    df_cleve = pd.read_csv(HEART_RAW_CLEVELAND, header=None, names=HEART_COLUMNS, na_values="?")
    df_hung = pd.read_csv(HEART_RAW_HUNGARIAN, header=None, names=HEART_COLUMNS, na_values="?")

    df = pd.concat([df_cleve, df_hung], ignore_index=True)

    # Clean numeric features
    for col in HEART_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binarize target 'num': 0 -> 0, 1,2,3,4 -> 1
    df = df.dropna(subset=["num"])
    df["num"] = (df["num"] > 0).astype(int)

    logger.info("Heart dataset loaded successfully. Shape: %s. Disease cases: %d, No-disease: %d",
                df.shape, (df["num"] == 1).sum(), (df["num"] == 0).sum())

    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_DIR / "heart_cleaned.csv", index=False)

    return df


if __name__ == "__main__":
    load_liver_data()
    load_ckd_data()
    load_diabetes_data()
    load_heart_data()
