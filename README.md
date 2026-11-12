# Multi-Disease Predictive Modeling and Diagnostic Support System

An end-to-end clinical machine learning system that predicts **Liver Disease**, **Chronic Kidney Disease (CKD)**, **Type-2 Diabetes**, and **Heart Disease** from routine laboratory biomarkers. The system trains and benchmarks four distinct model families, ensures strict zero-data-leakage preprocessing, explains individual and global predictions using **SHAP**, and serves interactive risk scoring in a multi-page **Streamlit** diagnostic application.

---

## 🩺 Clinical Overview & Problem Statement

Early detection of chronic disease prevents organ failure, reduces hospitalizations, and saves lives. However, primary care clinicians face high patient volumes and must mentally synthesize complex, multi-panel laboratory assays. 

Each condition presents unique data quirks, non-linear relationships, missing lab measurements, and class imbalance. This project delivers a production-grade diagnostic support platform that transforms raw clinical lab values into calibrated, explainable probability scores.

---

## 🔬 Four Model Families & Dataset Architecture

| Disease | Dataset | Algorithm Family | Primary Model | Benchmark Baseline | Key Clinical Features |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Liver** | Indian Liver Patient Dataset (ILPD) | **Bagging** | Random Forest | Logistic Regression | Bilirubin, SGPT (ALT), SGOT (AST), Alkaline Phosphatase, Albumin/Globulin ratio |
| **Kidney** | UCI Chronic Kidney Disease (CKD) | **Boosting** | XGBoost | LightGBM & Logistic Regression | Serum Creatinine, Blood Urea, Hemoglobin, Specific Gravity, Urinalysis |
| **Diabetes** | Pima Indians Diabetes Dataset | **Margin-based** | Support Vector Machine (RBF) | Logistic Regression | Fasting Glucose, Serum Insulin, BMI, Diabetes Pedigree Function, Age |
| **Heart** | Cleveland + Hungarian (UCI) | **Deep Learning** | Multi-Layer Perceptron (Keras) | Logistic Regression | Resting BP, Cholesterol, Thallium Stress Test, ST Depression, Chest Pain |

---

## 🛡️ Zero-Data-Leakage Pipeline Architecture

In medical machine learning, data leakage produces artificially inflated metrics that fail catastrophically in clinical deployment. This system implements rigorous safeguards:

1. **Stratified Split FIRST (80/20):** `train_test_split(..., stratify=y, random_state=42)` is executed strictly before any imputation, scaling, or oversampling.
2. **imblearn Cross-Validation Isolation:** Preprocessing steps and `SMOTE` oversampling are bundled into `imblearn.pipeline.Pipeline`. Synthetic samples and scaling parameters are computed strictly inside each cross-validation fold and never touch the validation or test partitions.
3. **Selective SMOTE Application:** SMOTE is only applied when the training class imbalance ratio exceeds 1.5:1 (e.g., Liver ~2.5:1, Diabetes ~1.87:1, CKD ~1.67:1). When classes are naturally balanced (Heart ~1.44:1), SMOTE is skipped and documented in logs.
4. **Clinical Threshold Calibration:** Instead of default `0.50` probability cuts, decision thresholds are tuned on the training distributions using $F_2$-score optimization to prioritize **Recall / Sensitivity** (minimizing life-threatening false negatives).

---

## 📁 Project Directory Structure

```text
├── data/
│   ├── raw/                 # Copies of raw source datasets
│   └── processed/           # Cleaned CSV datasets
├── Datasets/                # Original source datasets (untouched)
├── src/
│   ├── config.py            # Global constants, paths, and clinical feature lists
│   ├── data_loading.py      # Robust loaders handling dataset quirks
│   ├── preprocessing.py     # ColumnTransformers & imblearn pipelines
│   ├── train_liver.py       # Random Forest training & tuning
│   ├── train_kidney.py      # XGBoost & LightGBM training & tuning
│   ├── train_diabetes.py    # SVM RBF training & tuning
│   ├── train_heart.py       # Keras Multi-Layer Perceptron training
│   ├── evaluate.py          # Metrics calculation & threshold tuning
│   ├── explain.py           # Tree & Kernel SHAP explainability
│   └── inference.py         # Real-time cached inference & validator
├── models/                  # Fitted pipelines, .keras models, metadata JSONs
├── reports/
│   └── figures/             # Saved ROC curves, confusion matrices, SHAP plots
├── app/
│   ├── Home.py              # Multi-page Streamlit landing page with glassmorphism
│   ├── ui_utils.py          # Gauge charts, SHAP plots, and clinical summaries
│   └── pages/
│       ├── 1_Liver_Disease.py
│       ├── 2_Kidney_Disease.py
│       ├── 3_Diabetes.py
│       ├── 4_Heart_Disease.py
│       └── 5_Model_Comparison.py
├── assets/
│   ├── doctor_bg.jpg        # Dark-tinted landing page background
│   ├── style.css            # Custom CSS styling and badges
│   └── *.svg                # High-resolution organ icons
├── docs/
│   └── data_audit.md        # Step 0 comprehensive dataset audit
├── tests/                   # Pytest suite verifying loaders, pipelines, inference
├── requirements.txt         # Pinned production dependencies
├── train_all.py             # Master script to train all 4 diseases end-to-end
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
* **OS:** Windows 10/11, Linux, or macOS
* **Python:** Python **3.11** (64-bit) recommended for complete TensorFlow, XGBoost, and scikit-learn wheel compatibility.

### 2. Environment Setup
```powershell
# Clone or navigate to the project directory
cd "d:\Semester 3\Artifical Intellignece\Project"

# Create a virtual environment using Python 3.11
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install pinned dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Train & Evaluate All Models

Run the master training pipeline to execute 5-fold cross-validation, hyperparameter grid search, threshold tuning, metric computation, SHAP plot generation, and artifact serialization:

```powershell
python train_all.py
```

To train an individual disease model:
```powershell
python src/train_liver.py
python src/train_kidney.py
python src/train_diabetes.py
python src/train_heart.py
```

---

## 🧪 Running the Test Suite

Run `pytest` to verify that loaders parse schemas properly, no `NaN` values leak past transformers, train and test rows remain strictly disjoint, and real-time inference outputs valid probabilities:

```powershell
pytest tests/ -v
```

---

## 🖥️ Launching the Streamlit Application

Start the web application:

```powershell
streamlit run app/Home.py
```

* **Landing Page:** Interactive medical dashboard with dark-tinted overlay, clickable organ cards, and architecture summary.
* **Disease Pages:** Interactive patient lab input forms with sample case presets ("Low-Risk" and "High-Risk"), real-time probability gauge, clinical risk level badges, SHAP factor bar charts, and plain-language diagnostic summaries.
* **Model Comparison Page:** Comprehensive test benchmark table, interactive metric comparison charts, and deep-dive ROC curves, confusion matrices, and global SHAP summaries.

---

## 📊 Held-Out Test Set Results Summary

All metrics are evaluated on the untouched 20% stratified test sets. Decision thresholds were tuned using $F_2$-score optimization to prioritize recall/sensitivity in clinical screening.

| Disease | Model Family | Algorithm | Accuracy | Recall (Sensitivity) | Precision | F1-Score | ROC-AUC | Decision Threshold |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Liver** | Bagging | **Random Forest** | **70.09%** | **98.80%** | 70.69% | 0.8241 | 0.7899 | 0.10 |
| *Liver Baseline* | Linear | Logistic Regression | 71.79% | 65.06% | 93.10% | 0.7660 | 0.8359 | 0.50 |
| **Kidney** | Boosting | **XGBoost** | **100.0%** | **100.0%** | 100.0% | 1.0000 | 1.0000 | 0.40 |
| *Kidney Baseline* | Linear | Logistic Regression | 98.75% | 98.00% | 100.0% | 0.9899 | 0.9993 | 0.50 |
| **Diabetes** | Margin-based | **SVM (RBF Kernel)** | **66.88%** | **96.30%** | 51.49% | 0.6710 | 0.8076 | 0.21 |
| *Diabetes Baseline* | Linear | Logistic Regression | 70.78% | 64.81% | 57.38% | 0.6087 | 0.8113 | 0.50 |
| **Heart** | Deep Learning | **Keras MLP** | **80.83%** | **91.84%** | 70.31% | 0.7965 | 0.9023 | 0.24 |
| *Heart Baseline* | Linear | Logistic Regression | 82.50% | 77.55% | 79.17% | 0.7835 | 0.9074 | 0.50 |

---

## 🔍 Assumptions & Design Decisions

1. **Diabetes Dataset:** The Pima Indians dataset was provided as `pima-indians-diabetes.data.csv` (768 records, 9 columns). Clinically impossible zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing values (`NaN`) and imputed with the training median.
2. **CKD Trailing Commas & Whitespace:** Rows 69 and 72 in `chronic_kidney_disease.arff` contained trailing commas (`ckd,`). Values with stray `\t` characters were cleaned and cast to float/categorical.
3. **Heart Disease Target Binarization:** In accordance with clinical literature on the Cleveland and Hungarian datasets, angiographic narrowing values $0$ represent absence of disease, while values $1, 2, 3, 4$ represent presence of cardiovascular disease ($num > 0 \rightarrow 1$).
4. **Clinical Threshold Tuning:** Tuned on training fold distributions to achieve at least 80–90% sensitivity ($F_2$-score optimization), favoring recall over precision in early screening scenarios.

---

## ⚠️ Clinical Disclaimer

> **Educational project. Not a medical diagnosis tool.**  
> This software is intended solely for academic research and educational demonstration. Predictions should never replace clinical judgment, comprehensive medical history, or confirmatory diagnostic procedures by licensed healthcare professionals.
