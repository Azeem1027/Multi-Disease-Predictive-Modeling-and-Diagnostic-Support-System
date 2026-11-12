"""
Unit Tests for Data Loaders Across All Four Diseases.
"""
import pytest
import pandas as pd
from src.data_loading import (
    load_liver_data,
    load_ckd_data,
    load_diabetes_data,
    load_heart_data,
)
from src.config import (
    LIVER_COLUMNS,
    CKD_NUMERIC_FEATURES,
    CKD_CATEGORICAL_FEATURES,
    DIABETES_COLUMNS,
    HEART_COLUMNS,
)


def test_liver_loader():
    df = load_liver_data(save_processed=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (583, 11)
    assert list(df.columns) == LIVER_COLUMNS
    assert set(df["Dataset"].unique()).issubset({0, 1})
    assert set(df["Gender"].unique()).issubset({"Male", "Female"})


def test_ckd_loader():
    df = load_ckd_data(save_processed=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (400, 25)
    expected_cols = set(CKD_NUMERIC_FEATURES + CKD_CATEGORICAL_FEATURES + ["class"])
    assert set(df.columns) == expected_cols
    assert set(df["class"].unique()).issubset({0, 1})


def test_diabetes_loader():
    df = load_diabetes_data(save_processed=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (768, 9)
    assert list(df.columns) == DIABETES_COLUMNS
    assert set(df["Outcome"].unique()).issubset({0, 1})


def test_heart_loader():
    df = load_heart_data(save_processed=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (597, 14)
    assert list(df.columns) == HEART_COLUMNS
    assert set(df["num"].unique()).issubset({0, 1})

