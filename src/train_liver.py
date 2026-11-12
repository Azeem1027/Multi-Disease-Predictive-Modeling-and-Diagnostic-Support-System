"""
Training Script for Liver Disease (ILPD).
Model: Random Forest (Bagging) + Logistic Regression Baseline.
Handles ILPD quirks: Gender encoding, Albumin_and_Globulin_Ratio nulls, label mapping.
Applies imblearn Pipeline with SMOTE strictly inside CV folds.
"""
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from src.config import (
    LIVER_NUMERIC_FEATURES,
    LIVER_CATEGORICAL_FEATURES,
    LIVER_TARGET,
    RANDOM_SEED,
    CV_SPLITS,
    MODELS_DIR,
)
from src.data_loading import load_liver_data
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


def train_liver():
    logger.info("=" * 60)
    logger.info("STARTING LIVER DISEASE MODEL TRAINING (Random Forest)")
    logger.info("=" * 60)

    # 1. Load Data
    df = load_liver_data()

    # 2. Stratified train/test split FIRST (zero data leakage)
    X_train, X_test, y_train, y_test = get_train_test_data(df, target_col=LIVER_TARGET)

    # 3. Check SMOTE necessity
    use_smote, smote_reason = check_and_log_smote_need(y_train)

    # 4. Build feature preprocessor
    # For Random Forest, tree-based models don't strictly require scaling,
    # but we apply scaling in the pipeline for consistency and for linear baseline
    preprocessor = build_feature_preprocessor(
        numeric_features=LIVER_NUMERIC_FEATURES,
        categorical_features=LIVER_CATEGORICAL_FEATURES,
        scale_numeric=True
    )

    # 5. Build Random Forest imblearn Pipeline
    rf_base = RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced")
    pipeline = create_full_pipeline(preprocessor, rf_base, apply_smote=use_smote)

    # 6. Grid Search CV on training set only
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [5, 10, None],
        "model__min_samples_split": [2, 5],
    }
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_pipeline = grid_search.best_estimator_
    logger.info("Best RF Params: %s (Best CV F1: %.4f)", grid_search.best_params_, grid_search.best_score_)

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
    logger.info("Liver Baseline (Logistic Regression) Metrics: %s", baseline_metrics)

    # 9. Plot ROC and Confusion Matrix
    plot_and_save_roc_curve(y_test.values, y_test_prob, "liver", tuned_metrics["roc_auc"])
    plot_and_save_confusion_matrix(tuned_metrics["confusion_matrix"], "liver", optimal_thresh)

    # 10. Global SHAP Summary
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_rf = best_pipeline.named_steps["model"]
    X_train_trans = fitted_preprocessor.transform(X_train)
    X_test_trans = fitted_preprocessor.transform(X_test)
    
    # Extract feature names from preprocessor
    num_names = LIVER_NUMERIC_FEATURES
    cat_names = list(fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(LIVER_CATEGORICAL_FEATURES))
    feature_names = num_names + cat_names

    try:
        generate_global_shap_summary(
            model=fitted_rf,
            X_train_trans=X_train_trans,
            X_test_trans=X_test_trans,
            feature_names=feature_names,
            disease_name="liver",
            explainer_type="tree"
        )
    except Exception as e:
        logger.warning("SHAP generation encountered issue: %s", e)

    # 11. Calculate feature ranges for UI validation
    feature_ranges = {}
    for col in LIVER_NUMERIC_FEATURES:
        feature_ranges[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "median": float(df[col].median()),
            "type": "numeric"
        }
    feature_ranges["Gender"] = {
        "options": ["Male", "Female"],
        "default": "Male",
        "type": "categorical"
    }

    # 12. Save Models & Metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "liver_rf_pipeline.pkl"
    joblib.dump(best_pipeline, model_path)
    joblib.dump(lr_pipeline, MODELS_DIR / "liver_baseline.pkl")

    # Save a small background matrix for local patient SHAP explanations
    joblib.dump({
        "background": X_train_trans[:40],
        "feature_names": feature_names
    }, MODELS_DIR / "liver_shap_bg.pkl")

    metadata = {
        "disease": "liver",
        "primary_model": "Random Forest",
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
    save_disease_metadata("liver", metadata)
    logger.info("LIVER TRAINING COMPLETED SUCCESSFULLY. Tuned Test Metrics: %s", tuned_metrics)
    return metadata


if __name__ == "__main__":
    train_liver()

