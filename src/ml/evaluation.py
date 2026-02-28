from typing import Tuple, Any
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
    """
    Evaluates the model using RMSE and R² metrics.
    Args:
        model (Any): Trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test targets.
    Returns:
        Tuple[float, float]: RMSE and R² score.
    """
    if X_test.size == 0 or y_test.size == 0:
        raise ValueError("Test data cannot be empty.")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    n, p = X_test.shape
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    return rmse, r2, adj_r2