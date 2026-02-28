# src/ml/model.py

from sklearn.linear_model import LinearRegression
import numpy as np


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Train linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model