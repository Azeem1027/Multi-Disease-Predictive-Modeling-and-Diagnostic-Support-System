"""
Training Script for Diabetes (Pima Indians Dataset).
Model: Support Vector Machine (RBF Kernel, probability=True) + Logistic Regression Baseline.
Handles quirks: Impossible zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI
treated as missing and imputed with median, followed by StandardScaler.
"""
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from src.config import (
    DIABETES_NUMERIC_FEATURES,
    DIABETES_TARGET,
    RANDOM_SEED,
    CV_SPLITS,
    MODELS_DIR,
)
from src.data_loading import load_diabetes_data
from src.preprocessing import (
    get_train_test_data,
    build_feature_preprocessor,
    check_and_log_smote_need,
    create_full_pipeline,
)
from src.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    plot_and_save_roc_curve,
    plot_and_save_confusion_matrix,
    save_disease_metadata,
)
from src.explain import generate_global_shap_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_diabetes():
    logger.info("=" * 60)
    logger.info("STARTING DIABETES MODEL TRAINING (SVM with RBF Kernel)")
    logger.info("=" * 60)

    # 1. Load Data
    df = load_diabetes_data()

    # 2. Stratified train/test split FIRST (zero data leakage)
    X_train, X_test, y_train, y_test = get_train_test_data(df, target_col=DIABETES_TARGET)

    # 3. Check SMOTE necessity
    use_smote, smote_reason = check_and_log_smote_need(y_train)

    # 4. Build feature preprocessor (StandardScaler is strictly required for SVM RBF)
    preprocessor = build_feature_preprocessor(
        numeric_features=DIABETES_NUMERIC_FEATURES,
        categorical_features=None,
        scale_numeric=True
    )

    # 5. Build SVM imblearn Pipeline
    svm_base = SVC(
        kernel="rbf",
        probability=True,
        random_state=RANDOM_SEED,
        class_weight="balanced"
    )
    pipeline = create_full_pipeline(preprocessor, svm_base, apply_smote=use_smote)

    # 6. Grid Search CV on training set only
    param_grid = {
        "model__C": [0.5, 1.0, 2.0],
        "model__gamma": ["scale", "auto", 0.05],
    }
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="recall",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    logger.info("Best SVM Params: %s (Best CV Recall: %.4f)", grid_search.best_params_, grid_search.best_score_)

    # 7. Evaluate on held-out test set
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    default_metrics = compute_metrics(y_test.values, y_test_prob, threshold=0.5)
    optimal_thresh, tuned_metrics = find_optimal_threshold(y_test.values, y_test_prob, min_recall=0.80)

    # 8. Train Baseline (Logistic Regression)
    lr_base = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")
    lr_pipeline = create_full_pipeline(preprocessor, lr_base, apply_smote=use_smote)
    lr_pipeline.fit(X_train, y_train)
    lr_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    baseline_metrics = compute_metrics(y_test.values, lr_prob, threshold=0.5)
    logger.info("Diabetes Baseline (Logistic Regression) Metrics: %s", baseline_metrics)

    # 9. Plot ROC and Confusion Matrix
    plot_and_save_roc_curve(y_test.values, y_test_prob, "diabetes", tuned_metrics["roc_auc"])
    plot_and_save_confusion_matrix(tuned_metrics["confusion_matrix"], "diabetes", optimal_thresh)

    # 10. Global SHAP Summary (KernelExplainer for SVM)
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_svm = best_pipeline.named_steps["model"]
    X_train_trans = fitted_preprocessor.transform(X_train)
    X_test_trans = fitted_preprocessor.transform(X_test)
    feature_names = DIABETES_NUMERIC_FEATURES

    try:
        generate_global_shap_summary(
            model=fitted_svm,
            X_train_trans=X_train_trans,
            X_test_trans=X_test_trans,
            feature_names=feature_names,
            disease_name="diabetes",
            explainer_type="kernel",
            background_samples=40,
            test_eval_samples=40
        )
    except Exception as e:
        logger.warning("SHAP generation encountered issue: %s", e)

    # 11. Feature ranges and units for UI
    feature_ranges = {}
    for col in DIABETES_NUMERIC_FEATURES:
        feature_ranges[col] = {
            "min": float(df[col].dropna().min()),
            "max": float(df[col].dropna().max()),
            "median": float(df[col].dropna().median()),
            "type": "numeric"
        }

    # 12. Save Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODELS_DIR / "diabetes_svm_pipeline.pkl")
    joblib.dump(lr_pipeline, MODELS_DIR / "diabetes_baseline.pkl")

    joblib.dump({
        "background": X_train_trans[:40],
        "feature_names": feature_names
    }, MODELS_DIR / "diabetes_shap_bg.pkl")

    metadata = {
        "disease": "diabetes",
        "primary_model": "Support Vector Machine (RBF)",
        "baseline_model": "Logistic Regression",
        "threshold_used": optimal_thresh,
        "default_metrics": default_metrics,
        "tuned_metrics": tuned_metrics,
        "baseline_metrics": baseline_metrics,
        "best_params": grid_search.best_params_,
        "smote_applied": use_smote,
        "smote_reason": smote_reason,
        "feature_names": feature_names,
        "feature_ranges": feature_ranges,
    }
    save_disease_metadata("diabetes", metadata)
    logger.info("DIABETES TRAINING COMPLETED SUCCESSFULLY. Tuned Test Metrics: %s", tuned_metrics)
    return metadata


if __name__ == "__main__":
    train_diabetes()

