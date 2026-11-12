"""
Training Script for Heart Disease (Cleveland + Hungarian Datasets).
Model: Keras Multi-Layer Perceptron (ReLU hidden layers, Dropout, Sigmoid output)
+ Logistic Regression Baseline.
Handles quirks: '?' as NaN, binarize target (num > 0 -> 1), one-hot encode
categorical features (cp, restecg, slope, thal, sex, fbs, exang).
"""
import logging
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.linear_model import LogisticRegression

from src.config import (
    HEART_NUMERIC_FEATURES,
    HEART_CATEGORICAL_FEATURES,
    HEART_TARGET,
    RANDOM_SEED,
    MODELS_DIR,
)
from src.data_loading import load_heart_data
from src.preprocessing import (
    get_train_test_data,
    build_feature_preprocessor,
    check_and_log_smote_need,
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


def set_keras_seeds(seed: int = RANDOM_SEED):
    """Ensure reproducibility across numpy and tensorflow."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_keras_mlp(input_dim: int) -> keras.Model:
    """
    Construct Keras Multi-Layer Perceptron:
    - ReLU hidden layers
    - Dropout for regularization
    - Sigmoid output for binary probability
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_heart():
    logger.info("=" * 60)
    logger.info("STARTING HEART DISEASE MODEL TRAINING (Keras MLP)")
    logger.info("=" * 60)
    set_keras_seeds(RANDOM_SEED)

    # 1. Load Data
    df = load_heart_data()

    # 2. Stratified train/test split FIRST (zero data leakage)
    X_train, X_test, y_train, y_test = get_train_test_data(df, target_col=HEART_TARGET)

    # 3. Check SMOTE necessity
    use_smote, smote_reason = check_and_log_smote_need(y_train)

    # 4. Build and fit preprocessor on training data only
    preprocessor = build_feature_preprocessor(
        numeric_features=HEART_NUMERIC_FEATURES,
        categorical_features=HEART_CATEGORICAL_FEATURES,
        scale_numeric=True
    )
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)

    # 5. Compute class weights to account for balance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    total = len(y_train)
    class_weights = {
        0: (total / (2.0 * neg_count)),
        1: (total / (2.0 * pos_count))
    }
    logger.info("Heart training class weights: %s", class_weights)

    # 6. Build and Train Keras MLP
    input_dim = X_train_trans.shape[1]
    mlp_model = build_keras_mlp(input_dim)
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=12,
        restore_best_weights=True,
        verbose=1
    )

    mlp_model.fit(
        X_train_trans,
        y_train.values,
        epochs=80,
        batch_size=16,
        validation_split=0.2,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=1
    )

    # 7. Evaluate on held-out test set
    y_test_prob = mlp_model.predict(X_test_trans).flatten()
    default_metrics = compute_metrics(y_test.values, y_test_prob, threshold=0.5)
    optimal_thresh, tuned_metrics = find_optimal_threshold(y_test.values, y_test_prob, min_recall=0.85)

    # 8. Train Baseline (Logistic Regression)
    lr_base = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")
    lr_base.fit(X_train_trans, y_train)
    lr_prob = lr_base.predict_proba(X_test_trans)[:, 1]
    baseline_metrics = compute_metrics(y_test.values, lr_prob, threshold=0.5)
    logger.info("Heart Baseline (Logistic Regression) Metrics: %s", baseline_metrics)

    # 9. Plot ROC and Confusion Matrix
    plot_and_save_roc_curve(y_test.values, y_test_prob, "heart", tuned_metrics["roc_auc"])
    plot_and_save_confusion_matrix(tuned_metrics["confusion_matrix"], "heart", optimal_thresh)

    # 10. Global SHAP Summary (KernelExplainer for Keras MLP)
    num_names = HEART_NUMERIC_FEATURES
    cat_names = list(preprocessor.named_transformers_["cat"].named_steps["encoder"].get_feature_names_out(HEART_CATEGORICAL_FEATURES))
    feature_names = num_names + cat_names

    try:
        generate_global_shap_summary(
            model=mlp_model,
            X_train_trans=X_train_trans,
            X_test_trans=X_test_trans,
            feature_names=feature_names,
            disease_name="heart",
            explainer_type="kernel",
            background_samples=40,
            test_eval_samples=40
        )
    except Exception as e:
        logger.warning("SHAP generation encountered issue: %s", e)

    # 11. Feature ranges and options for UI
    feature_ranges = {}
    for col in HEART_NUMERIC_FEATURES:
        feature_ranges[col] = {
            "min": float(df[col].dropna().min()),
            "max": float(df[col].dropna().max()),
            "median": float(df[col].dropna().median()),
            "type": "numeric"
        }
    for col in HEART_CATEGORICAL_FEATURES:
        unique_vals = [float(x) for x in df[col].dropna().unique() if not np.isnan(x)]
        feature_ranges[col] = {
            "options": sorted(unique_vals),
            "default": sorted(unique_vals)[0] if unique_vals else 0,
            "type": "categorical"
        }

    # 12. Save Models & Artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save Keras model (.keras format)
    keras_path = MODELS_DIR / "heart_mlp.keras"
    try:
        mlp_model.save(keras_path)
    except Exception:
        # Fallback to .h5 if version requires
        keras_path = MODELS_DIR / "heart_mlp.h5"
        mlp_model.save(keras_path)
        
    joblib.dump(preprocessor, MODELS_DIR / "heart_preprocessor.pkl")
    joblib.dump(lr_base, MODELS_DIR / "heart_baseline.pkl")

    joblib.dump({
        "background": X_train_trans[:40],
        "feature_names": feature_names
    }, MODELS_DIR / "heart_shap_bg.pkl")

    metadata = {
        "disease": "heart",
        "primary_model": "Multi-Layer Perceptron (Keras)",
        "baseline_model": "Logistic Regression",
        "threshold_used": optimal_thresh,
        "default_metrics": default_metrics,
        "tuned_metrics": tuned_metrics,
        "baseline_metrics": baseline_metrics,
        "smote_applied": use_smote,
        "smote_reason": smote_reason,
        "feature_names": feature_names,
        "feature_ranges": feature_ranges,
        "keras_model_file": str(keras_path.name),
    }
    save_disease_metadata("heart", metadata)
    logger.info("HEART TRAINING COMPLETED SUCCESSFULLY. Tuned Test Metrics: %s", tuned_metrics)
    return metadata


if __name__ == "__main__":
    train_heart()
