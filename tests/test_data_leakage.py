"""
Unit Tests for Data Leakage Prevention.
Verifies that test rows never appear in the training partition across all datasets.
"""
import pytest
from src.data_loading import load_liver_data, load_ckd_data, load_diabetes_data, load_heart_data
from src.preprocessing import get_train_test_data
from src.config import LIVER_TARGET, CKD_TARGET, DIABETES_TARGET, HEART_TARGET


def test_liver_no_leakage():
    df = load_liver_data(save_processed=False)
    X_train, X_test, y_train, y_test = get_train_test_data(df, LIVER_TARGET)
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert train_indices.isdisjoint(test_indices), "Train and Test indices overlap in Liver dataset!"


def test_ckd_no_leakage():
    df = load_ckd_data(save_processed=False)
    X_train, X_test, y_train, y_test = get_train_test_data(df, CKD_TARGET)
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert train_indices.isdisjoint(test_indices), "Train and Test indices overlap in CKD dataset!"


def test_diabetes_no_leakage():
    df = load_diabetes_data(save_processed=False)
    X_train, X_test, y_train, y_test = get_train_test_data(df, DIABETES_TARGET)
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert train_indices.isdisjoint(test_indices), "Train and Test indices overlap in Diabetes dataset!"


def test_heart_no_leakage():
    df = load_heart_data(save_processed=False)
    X_train, X_test, y_train, y_test = get_train_test_data(df, HEART_TARGET)
    train_indices = set(X_train.index)
    test_indices = set(X_test.index)
    assert train_indices.isdisjoint(test_indices), "Train and Test indices overlap in Heart dataset!"

