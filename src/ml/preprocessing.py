from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes non-numeric columns except for the target column 'life_expectancy'.

    Returns cleaned dataframe with only numeric columns and 'life_expectancy'.
    """
    if 'life_expectancy' not in df.columns:
        raise ValueError("Target column 'life_expectancy' not found in dataframe.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'life_expectancy' not in numeric_cols:
        numeric_cols.append('life_expectancy')
    return df[numeric_cols].copy()

def handle_missing_values(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    """
    Handles missing values. Supports mean or KNN imputation.
    Args:
        df (pd.DataFrame): Input dataframe.
        method (str): "mean" or "knn".
    Returns Dataframe with imputed values.
    """

    if df.empty:
        raise ValueError("Input dataframe is empty.")

    # detect target if exists
    target_col = "life_expectancy" if "life_expectancy" in df.columns else None

    # split features + target
    if target_col:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        X = df.copy()

    #  MEAN imputation 
    if method == "mean":
        X_imputed = X.fillna(X.mean())

    #  KNN imputation 
    elif method == "knn":
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

    else:
        raise ValueError("Imputation method must be 'mean' or 'knn'.")

    # restore dataframe structure
    if target_col:
        return pd.concat([X_imputed, y], axis=1)
    else:
        return X_imputed

def remove_outliers_iqr(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Removes rows with outliers based on IQR for all numeric columns except target.
    Returns Dataframe with outliers removed.
    """
    numeric_cols = [col for col in df.columns if col != target and pd.api.types.is_numeric_dtype(df[col])]
    mask = pd.Series(True, index=df.index)
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= df[col].between(lower, upper)
    return df[mask].copy()

def split_data(df: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataframe into train and test sets.
    Args:
        df (pd.DataFrame): Input dataframe.
        target (str): Target column name.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataframe.")
    X = df.drop(columns=[target]).values
    y = df[target].values
    return train_test_split(X, y, test_size=0.2, random_state=42)