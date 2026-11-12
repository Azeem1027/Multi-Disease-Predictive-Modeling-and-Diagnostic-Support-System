import csv
import re
from collections import Counter

print("=== 1. ILPD AUDIT ===")
ilpd_cols = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
    'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
    'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset'
]
with open('data/raw/ilpd/Indian Liver Patient Dataset (ILPD).csv', 'r') as f:
    rows = list(csv.reader(f))
print(f"Shape: ({len(rows)}, {len(rows[0]) if rows else 0})")
missing = {col: 0 for col in ilpd_cols}
classes = Counter()
for r in rows:
    for col, val in zip(ilpd_cols, r):
        val = val.strip()
        if val == '' or val == '?':
            missing[col] += 1
    classes[r[-1].strip()] += 1
print("Missing counts:", missing)
print("Class balance (1=patient, 2=non-patient):", dict(classes))

print("\n=== 2. CKD AUDIT ===")
with open('data/raw/ckd/Chronic_Kidney_Disease/chronic_kidney_disease.arff', 'r') as f:
    lines = [l.strip() for l in f if l.strip()]
ckd_attrs = []
ckd_data = []
is_data = False
for l in lines:
    if l.lower().startswith('@attribute'):
        parts = l.split()
        ckd_attrs.append(parts[1].replace("'", ""))
    elif l.lower().startswith('@data'):
        is_data = True
    elif is_data:
        ckd_data.append([x.strip().strip("'") for x in l.split(',')])

print(f"Shape: ({len(ckd_data)}, {len(ckd_attrs)})")
print("Attributes:", ckd_attrs)
ckd_missing = {attr: 0 for attr in ckd_attrs}
ckd_classes = Counter()
for r in ckd_data:
    for attr, val in zip(ckd_attrs, r):
        val_clean = val.strip().replace('\t', '')
        if val_clean in ['', '?']:
            ckd_missing[attr] += 1
    target = r[-1].strip().replace('\t', '')
    ckd_classes[target] += 1
print("Missing counts:", ckd_missing)
print("Class balance:", dict(ckd_classes))

print("\n=== 3. HEART AUDIT ===")
heart_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
with open('data/raw/heart/processed.cleveland.data', 'r') as f:
    cleve = [[x.strip() for x in l.strip().split(',')] for l in f if l.strip()]
with open('data/raw/heart/processed.hungarian.data', 'r') as f:
    hung = [[x.strip() for x in l.strip().split(',')] for l in f if l.strip()]

print(f"Cleveland Shape: ({len(cleve)}, {len(cleve[0]) if cleve else 0})")
print(f"Hungarian Shape: ({len(hung)}, {len(hung[0]) if hung else 0})")
combined = cleve + hung
print(f"Combined Heart Shape: ({len(combined)}, {len(combined[0]) if combined else 0})")

heart_missing = {col: 0 for col in heart_cols}
heart_classes = Counter()
for r in combined:
    for col, val in zip(heart_cols, r):
        if val in ['', '?']:
            heart_missing[col] += 1
    target = r[-1]
    bin_target = 1 if (target != '?' and int(float(target)) > 0) else 0
    heart_classes[bin_target] += 1
print("Missing counts (Combined):", heart_missing)
print("Binarized Class balance (0=No disease, 1=Disease):", dict(heart_classes))

print("\n=== 4. DIABETES AUDIT ===")
print("Checking Datasets/diabetes...")
try:
    with open('data/raw/diabetes/README', 'r') as f:
        content = f.read(200)
        print("README header:", content.replace('\n', ' '))
except Exception as e:
    print("Error reading README:", e)

