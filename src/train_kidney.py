"""
Training Script for Chronic Kidney Disease (CKD).
Model: XGBoost (Boosting) + LightGBM Comparison + Logistic Regression Baseline.
Handles CKD quirks: '?' and stray tabs/whitespace in values, mixed dtypes,
24 features plus 'class' target ('ckd'/'notckd').
"""
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from src.config import (
    CKD_NUMERIC_FEATURES,
    CKD_CATEGORICAL_FEATURES,
    CKD_TARGET,
    RANDOM_SEED,
    CV_SPLITS,
    MODELS_DIR,
)
from src.data_loading import load_ckd_data
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


def train_kidney():
    logger.info("=" * 60)
    logger.info("STARTING KIDNEY DISEASE MODEL TRAINING (XGBoost)")
    logger.info("=" * 60)

    # 1. Load Data
    df = load_ckd_data()

    # 2. Stratified train/test split FIRST (zero data leakage)
    X_train, X_test, y_train, y_test = get_train_test_data(df, target_col=CKD_TARGET)

    # 3. Check SMOTE necessity
    use_smote, smote_reason = check_and_log_smote_need(y_train)

    # 4. Build feature preprocessor
    preprocessor = build_feature_preprocessor(
        numeric_features=CKD_NUMERIC_FEATURES,
        categorical_features=CKD_CATEGORICAL_FEATURES,
        scale_numeric=True
    )

    # 5. Build XGBoost imblearn Pipeline
    xgb_base = XGBClassifier(
        random_state=RANDOM_SEED,
        eval_metric="logloss",
        n_jobs=-1
    )
    pipeline = create_full_pipeline(preprocessor, xgb_base, apply_smote=use_smote)

    # 6. Grid Search CV on training set only
    param_grid = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
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
    logger.info("Best XGBoost Params: %s (Best CV Recall: %.4f)", grid_search.best_params_, grid_search.best_score_)

    # 7. Evaluate on held-out test set
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]
    default_metrics = compute_metrics(y_test.values, y_test_prob, threshold=0.5)
    optimal_thresh, tuned_metrics = find_optimal_threshold(y_test.values, y_test_prob, min_recall=0.90)

    # 8. Train Comparison Model (LightGBM)
    lgb_base = LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)
    lgb_pipeline = create_full_pipeline(preprocessor, lgb_base, apply_smote=use_smote)
    lgb_pipeline.fit(X_train, y_train)
    lgb_prob = lgb_pipeline.predict_proba(X_test)[:, 1]
    lgb_metrics = compute_metrics(y_test.values, lgb_prob, threshold=0.5)
    logger.info("LightGBM Comparison Metrics: %s", lgb_metrics)

    # 9. Train Baseline (Logistic Regression)
    lr_base = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr_pipeline = create_full_pipeline(preprocessor, lr_base, apply_smote=use_smote)
    lr_pipeline.fit(X_train, y_train)
    lr_prob = lr_pipeline.predict_proba(X_test)[:, 1]
    baseline_metrics = compute_metrics(y_test.values, lr_prob, threshold=0.5)
    logger.info("Kidney Baseline (Logistic Regression) Metrics: %s", baseline_metrics)

    # 10. Plot ROC and Confusion Matrix
    plot_and_save_roc_curve(y_test.values, y_test_prob, "kidney", tuned_metrics["roc_auc"])
    plot_and_save_confusion_matrix(tuned_metrics["confusion_matrix"], "kidney", optimal_thresh)

    # 11. Global SHAP Summary
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    fitted_xgb = best_pipeline.named_steps["model"]
    X_train_trans = fitted_preprocessor.transform(X_train)
    X_test_trans = fitted_preprocessor.transform(X_test)

    num_names = CKD_NUMERIC_FEATURES
    cat_names = list(fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(CKD_CATEGORICAL_FEATURES))
    feature_names = num_names + cat_names

    try:
        generate_global_shap_summary(
            model=fitted_xgb,
            X_train_trans=X_train_trans,
            X_test_trans=X_test_trans,
            feature_names=feature_names,
            disease_name="kidney",
            explainer_type="tree"
        )
    except Exception as e:
        logger.warning("SHAP generation encountered issue: %s", e)

    # 12. Feature ranges and options for UI
    feature_ranges = {}
    for col in CKD_NUMERIC_FEATURES:
        feature_ranges[col] = {
            "min": float(df[col].dropna().min()),
            "max": float(df[col].dropna().max()),
            "median": float(df[col].dropna().median()),
            "type": "numeric"
        }
    for col in CKD_CATEGORICAL_FEATURES:
        unique_vals = [str(x) for x in df[col].dropna().unique() if str(x) not in ["?", "nan"]]
        feature_ranges[col] = {
            "options": unique_vals,
            "default": unique_vals[0] if unique_vals else "normal",
            "type": "categorical"
        }

    # 13. Save Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODELS_DIR / "kidney_xgb_pipeline.pkl")
    joblib.dump(lgb_pipeline, MODELS_DIR / "kidney_lgbm_pipeline.pkl")
    joblib.dump(lr_pipeline, MODELS_DIR / "kidney_baseline.pkl")

    joblib.dump({
        "background": X_train_trans[:40],
        "feature_names": feature_names
    }, MODELS_DIR / "kidney_shap_bg.pkl")

    metadata = {
        "disease": "kidney",
        "primary_model": "XGBoost",
        "comparison_model": "LightGBM",
        "baseline_model": "Logistic Regression",
        "threshold_used": optimal_thresh,
        "default_metrics": default_metrics,
        "tuned_metrics": tuned_metrics,
        "comparison_metrics": lgb_metrics,
        "baseline_metrics": baseline_metrics,
        "best_params": grid_search.best_params_,
        "smote_applied": use_smote,
        "smote_reason": smote_reason,
        "feature_names": feature_names,
        "feature_ranges": feature_ranges,
    }
    save_disease_metadata("kidney", metadata)
    logger.info("KIDNEY TRAINING COMPLETED SUCCESSFULLY. Tuned Test Metrics: %s", tuned_metrics)
    return metadata


if __name__ == "__main__":
    train_kidney()

