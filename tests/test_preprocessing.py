"""
Unit Tests for Preprocessing Pipelines.
Verifies no NaNs remain after transformation.
"""
import pytest
import numpy as np
from src.data_loading import load_liver_data, load_ckd_data, load_diabetes_data, load_heart_data
from src.preprocessing import build_feature_preprocessor, get_train_test_data
from src.config import (
    LIVER_NUMERIC_FEATURES,
    LIVER_CATEGORICAL_FEATURES,
    LIVER_TARGET,
    CKD_NUMERIC_FEATURES,
    CKD_CATEGORICAL_FEATURES,
    CKD_TARGET,
    DIABETES_NUMERIC_FEATURES,
    DIABETES_TARGET,
    HEART_NUMERIC_FEATURES,
    HEART_CATEGORICAL_FEATURES,
    HEART_TARGET,
)


def test_liver_preprocessing_no_nan():
    df = load_liver_data(save_processed=False)
    X_train, X_test, _, _ = get_train_test_data(df, LIVER_TARGET)
    preprocessor = build_feature_preprocessor(LIVER_NUMERIC_FEATURES, LIVER_CATEGORICAL_FEATURES)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


def test_ckd_preprocessing_no_nan():
    df = load_ckd_data(save_processed=False)
    X_train, X_test, _, _ = get_train_test_data(df, CKD_TARGET)
    preprocessor = build_feature_preprocessor(CKD_NUMERIC_FEATURES, CKD_CATEGORICAL_FEATURES)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


def test_diabetes_preprocessing_no_nan():
    df = load_diabetes_data(save_processed=False)
    X_train, X_test, _, _ = get_train_test_data(df, DIABETES_TARGET)
    preprocessor = build_feature_preprocessor(DIABETES_NUMERIC_FEATURES, None)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()


def test_heart_preprocessing_no_nan():
    df = load_heart_data(save_processed=False)
    X_train, X_test, _, _ = get_train_test_data(df, HEART_TARGET)
    preprocessor = build_feature_preprocessor(HEART_NUMERIC_FEATURES, HEART_CATEGORICAL_FEATURES)
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    assert not np.isnan(X_train_trans).any()
    assert not np.isnan(X_test_trans).any()

