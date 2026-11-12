"""
Preprocessing, Train/Test Splitting, and imblearn Leak-Free Pipelines.
Ensures zero data leakage: splitting precedes all fitting, and SMOTE/scaling
occur strictly within CV folds or on training data only.
"""
import logging
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.config import RANDOM_SEED, TEST_SIZE

logger = logging.getLogger(__name__)


def get_train_test_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform Stratified Train/Test Split FIRST before any fitting or scaling.
    Guarantees no data leakage.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    logger.info("Stratified split completed. Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)
    train_dist = y_train.value_counts(normalize=True).to_dict()
    test_dist = y_test.value_counts(normalize=True).to_dict()
    logger.info("Class distributions - Train: %s | Test: %s", train_dist, test_dist)

    return X_train, X_test, y_train, y_test


def build_feature_preprocessor(
    numeric_features: List[str],
    categorical_features: Optional[List[str]] = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """
    Construct ColumnTransformer for imputation, optional scaling, and encoding.
    
    Rules:
    - Median imputation for numeric.
    - Most-frequent imputation for categorical.
    - StandardScaler applied if scale_numeric is True.
    - OneHotEncoder (handle_unknown='ignore') for categorical features.
    """
    transformers = []

    # Numeric Pipeline
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_transformer = ImbPipeline(steps=num_steps)
    transformers.append(("num", numeric_transformer, numeric_features))

    # Categorical Pipeline
    if categorical_features:
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]
        categorical_transformer = ImbPipeline(steps=cat_steps)
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def check_and_log_smote_need(y_train: pd.Series, ratio_threshold: float = 1.5) -> Tuple[bool, str]:
    """
    Determine if SMOTE is warranted based on training class ratio.
    Provides clear justification in logs.
    """
    counts = y_train.value_counts()
    if len(counts) < 2:
        return False, "Single class present in training data."
    
    maj_class = counts.max()
    min_class = counts.min()
    ratio = maj_class / min_class

    if ratio >= ratio_threshold:
        reason = f"Class ratio is {ratio:.2f}:1 (exceeds {ratio_threshold}:1 threshold). SMOTE applied in training CV."
        logger.info("[SMOTE Decision] %s", reason)
        return True, reason
    else:
        reason = f"Class ratio is {ratio:.2f}:1 (below {ratio_threshold}:1 threshold, already balanced). SMOTE skipped."
        logger.info("[SMOTE Decision] %s", reason)
        return False, reason


def create_full_pipeline(
    preprocessor: ColumnTransformer,
    model: Any,
    apply_smote: bool = True,
    random_state: int = RANDOM_SEED
) -> ImbPipeline:
    """
    Create an imblearn Pipeline so that SMOTE, imputation, and scaling happen
    strictly inside each CV fold and never leak onto test data.
    """
    steps = [("preprocessor", preprocessor)]
    if apply_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("model", model))

    return ImbPipeline(steps=steps)

