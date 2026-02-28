from typing import Any
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """
    Trains a Linear Regression model.
    Args:
        X_train : Training features.
        y_train: Training targets.
    Return Trained LinearRegression model.
    Raises ValueError if input arrays are empty or shapes are invalid.
    """
    if X_train.size == 0 or y_train.size == 0:
        raise ValueError("Training data cannot be empty.")
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError("X_train and y_train must have the same number of samples.")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    model.fit(X_train, y_train)
    return model