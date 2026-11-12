# Data Audit Report

**Project:** Multi-Disease Predictive Modeling and Diagnostic Support System  
**Audit Date:** 2026-11-07  
**Location Audited:** `D:\Semester 3\Artifical Intellignece\Project\Datasets\`

---

## 1. Indian Liver Patient Dataset (ILPD)

- **Source File:** `Datasets/ilpd+indian+liver+patient+dataset/Indian Liver Patient Dataset (ILPD).csv`
- **Extracted Copy:** `data/raw/ilpd/Indian Liver Patient Dataset (ILPD).csv`
- **Shape:** `(583, 11)`
- **Target Variable:** `Dataset` (Column 11)
  - Raw values: `1` (Liver Patient), `2` (Non-Patient)
  - Target mapping: `1 -> 1` (Positive / Patient), `2 -> 0` (Negative / Non-Patient)
  - Class Distribution:
    - Class 1 (Patient): 416 (71.36%)
    - Class 2 (Non-Patient): 167 (28.64%)
    - Class Imbalance Ratio: ~2.5 : 1
- **Features & Data Types:**
  1. `Age`: Integer (Range: 4 - 90)
  2. `Gender`: Categorical (`Male`: 441, `Female`: 142)
  3. `Total_Bilirubin`: Float (Range: 0.4 - 75.0)
  4. `Direct_Bilirubin`: Float (Range: 0.1 - 19.7)
  5. `Alkaline_Phosphotase`: Integer (Range: 63 - 2110)
  6. `Alamine_Aminotransferase` (SGPT): Integer (Range: 10 - 2000)
  7. `Aspartate_Aminotransferase` (SGOT): Integer (Range: 10 - 4929)
  8. `Total_Protiens`: Float (Range: 2.7 - 9.6)
  9. `Albumin`: Float (Range: 0.9 - 5.5)
  10. `Albumin_and_Globulin_Ratio`: Float (Range: 0.3 - 2.8)
- **Missing Values:**
  - `Albumin_and_Globulin_Ratio`: 4 missing values (0.69%)
  - All other features: 0 missing values
- **Quirks & Preprocessing Notes:**
  - `Gender` requires binary encoding (`Female: 0, Male: 1`).
  - Missing `Albumin_and_Globulin_Ratio` will be imputed with median.
  - Skewed enzyme distributions (SGPT, SGOT, Alkphos) handled effectively by Random Forest.
  - Pipeline will use SMOTE inside cross-validation folds due to the 2.5:1 imbalance.

---

## 2. Chronic Kidney Disease (CKD)

- **Source File:** `Datasets/chronic+kidney+disease/chronic+kidney_disease.zip` -> `Chronic_Kidney_Disease.rar`
- **Extracted Copy:** `data/raw/ckd/Chronic_Kidney_Disease/chronic_kidney_disease.arff`
- **Shape:** `(400, 25)`
- **Target Variable:** `class` (Column 25)
  - Raw values: `'ckd'`, `'notckd'` (with whitespace/tabs and trailing commas in rows 69 and 72: `'ckd,'`)
  - Cleaned Distribution:
    - `ckd` (Positive): 250 (62.5%)
    - `notckd` (Negative): 150 (37.5%)
    - Class Imbalance Ratio: ~1.67 : 1
- **Features (24 Predictors):**
  - **Numerical (11):** `age`, `bp` (blood pressure), `bgr` (blood glucose random), `bu` (blood urea), `sc` (serum creatinine), `sod` (sodium), `pot` (potassium), `hemo` (hemoglobin), `pcv` (packed cell volume), `wbcc` (white blood cell count), `rbcc` (red blood cell count).
  - **Nominal / Categorical (13):** `sg` (specific gravity), `al` (albumin), `su` (sugar), `rbc` (red blood cells), `pc` (pus cell), `pcc` (pus cell clumps), `ba` (bacteria), `htn` (hypertension), `dm` (diabetes mellitus), `cad` (coronary artery disease), `appet` (appetite), `pe` (pedal edema), `ane` (anemia).
- **Missing Values (Represented as `?` or whitespace):**
  - `age`: 9 | `bp`: 12 | `sg`: 47 | `al`: 46 | `su`: 49
  - `rbc`: 152 | `pc`: 65 | `pcc`: 4 | `ba`: 4
  - `bgr`: 44 | `bu`: 19 | `sc`: 17 | `sod`: 87 | `pot`: 88
  - `hemo`: 52 | `pcv`: 71 | `wbcc`: 106 | `rbcc`: 131
  - `htn`: 2 | `dm`: 3 | `cad`: 2 | `appet`: 1 | `pe`: 1 | `ane`: 1
- **Quirks & Preprocessing Notes:**
  - Values contain tabs (`\t`), trailing commas, leading/trailing whitespace.
  - Features like `pcv`, `wbcc`, `rbcc` are typed as strings in raw ARFF due to `?` and stray `\t` characters; require casting to float after cleaning.
  - Categorical strings contain typos (e.g., `' yes'`, `'\tno'`, `'ckd\t'`).
  - Preprocessor: Median imputation for numeric, most-frequent imputation for nominal, one-hot/ordinal encoding for nominals, followed by XGBoost / LightGBM.

---

## 3. Heart Disease (Cleveland + Hungarian)

- **Source Files:** `Datasets/heart+disease/processed.cleveland.data`, `Datasets/heart+disease/processed.hungarian.data`
- **Extracted Copies:** `data/raw/heart/processed.cleveland.data`, `data/raw/heart/processed.hungarian.data`
- **Shape:**
  - Cleveland: `(303, 14)`
  - Hungarian: `(294, 14)`
  - Combined: `(597, 14)`
- **Target Variable:** `num` (Column 14)
  - Diagnosis of heart disease (angiographic disease status in any major vessel):
    - `0`: < 50% diameter narrowing (No disease)
    - `1, 2, 3, 4`: > 50% diameter narrowing (Disease present)
  - Binarized Target: `num > 0 -> 1` else `0`
  - Class Distribution (Combined):
    - Negative (`0`): 352 (58.96%)
    - Positive (`1`): 245 (41.04%)
    - Balanced distribution (~1.44 : 1)
- **Features (13 Predictors):**
  1. `age`: Age in years
  2. `sex`: Sex (1 = male; 0 = female)
  3. `cp`: Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic)
  4. `trestbps`: Resting blood pressure (mm Hg)
  5. `chol`: Serum cholesterol (mg/dl)
  6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
  7. `restecg`: Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy)
  8. `thalach`: Maximum heart rate achieved
  9. `exang`: Exercise induced angina (1 = yes; 0 = no)
  10. `oldpeak`: ST depression induced by exercise relative to rest
  11. `slope`: Slope of peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping)
  12. `ca`: Number of major vessels colored by flourosopy (0 - 3)
  13. `thal`: Thallium scan (3 = normal, 6 = fixed defect, 7 = reversible defect)
- **Missing Values (`?`):**
  - `trestbps`: 1 | `chol`: 23 | `fbs`: 8 | `restecg`: 1 | `thalach`: 1 | `exang`: 1
  - `slope`: 190 | `ca`: 295 | `thal`: 268
- **Quirks & Preprocessing Notes:**
  - High missing rate in `ca`, `thal`, and `slope` in the Hungarian cohort.
  - Imputation: Median for numeric, most-frequent for categorical.
  - One-hot encoding for multi-class categorical features (`cp`, `restecg`, `slope`, `thal`).
  - Standard scaling required for neural network (MLP) input.

---

## 4. Diabetes Dataset (CRITICAL AUDIT FINDING)

- **Source File:** `Datasets/diabetes/` (`diabetes-data.tar.Z`, `Index`, `README`)
- **Extracted Copy:** `data/raw/diabetes/`
- **Contents Inspected:**
  - `README`: Mentions "prepared for the use of participants for the 1994 AAAI Spring Symposium on Artificial Intelligence in Medicine. * diabetes-data.tar.Z contains the distribution for 70 sets of data recorded on diabetes patients (several weeks' to months' worth of glucose, insulin, and lifestyle data per patient...)"
  - Archive contains: 70 individual patient time-series files (`data-01` to `data-70`), `Data-Codes`, `Domain-Description`.
- **Finding:**
  - This is the **UCI 1994 Time-Series Diabetes dataset**, NOT the tabular **Pima Indians Diabetes Dataset**.
  - The required Pima Indians schema is:
    `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome` (768 rows, 9 columns).
- **Project Rule Triggered:**
  > *"CHECK: the "diabetes" folder must be the Pima Indians dataset (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome). If it is instead the UCI time-series diabetes data (many per-patient files with date/time/code/value), STOP and tell me to supply the Pima CSV. Do not fake or substitute data."*
- **Action Required:** Execution paused as requested. The user must provide `diabetes.csv` (Pima Indians dataset).

