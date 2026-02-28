# src/ml/preprocessing.py

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def clean_data(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Keeps only numeric columns and target.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target not in numeric_cols:
        raise ValueError(f"Target column '{target}' must be numeric.")
    return df[numeric_cols].copy()


def split_data(
    df: pd.DataFrame,
    target: str,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits dataset BEFORE preprocessing to avoid leakage.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=42)

def remove_outliers_iqr(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Removes outliers based ONLY on the target variable.
    """

    q1 = y_train.quantile(0.25)
    q3 = y_train.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = y_train.between(lower, upper)

    print("Train size BEFORE outlier removal:", len(y_train))
    print("Train size AFTER outlier removal:", mask.sum())

    return X_train[mask], y_train[mask]


def preprocess_training_data(
    X_train: pd.DataFrame
) -> Tuple[np.ndarray, SimpleImputer, StandardScaler]:
    """
    Fit imputer and scaler ONLY on training data.
    """
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_imputed = imputer.fit_transform(X_train)
    X_scaled = scaler.fit_transform(X_imputed)

    return X_scaled, imputer, scaler


def preprocess_test_data(
    X_test: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler
) -> np.ndarray:
    """
    Apply fitted preprocessing to test data.
    """
    X_imputed = imputer.transform(X_test)
    X_scaled = scaler.transform(X_imputed)
    return X_scaled