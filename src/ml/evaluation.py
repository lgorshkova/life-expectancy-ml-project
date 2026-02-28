# src/ml/evaluation.py

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def evaluate_model(model, X, y):
    """
    Returns RMSE and R².
    """
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return rmse, r2, y_pred


def adjusted_r2(r2, n, p):
    """
    Computes adjusted R².
    """
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)