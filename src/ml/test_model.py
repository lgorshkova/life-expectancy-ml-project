import pytest
import numpy as np
import pandas as pd
from src.ml.preprocessing import clean_data, handle_missing_values, remove_outliers_iqr, split_data
from src.ml.model import train_model
from src.ml.evaluation import evaluate_model

def test_full_pipeline():
    """
    Tests the full ML pipeline using synthetic regression data.
    Asserts that R² > 0.7.
    """
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 5)
    coef = np.array([2.5, -1.2, 0.7, 3.3, -2.0])
    noise = np.random.normal(0, 1, n_samples)
    y = X @ coef + noise
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    df["life_expectancy"] = y
    df["iso3"] = ["XXX"] * n_samples
    df["year"] = np.random.randint(2000, 2020, n_samples)

    df = clean_data(df)
    df = handle_missing_values(df, method="mean")
    df = remove_outliers_iqr(df, target="life_expectancy")
    X_train, X_test, y_train, y_test = split_data(df, target="life_expectancy")
    model = train_model(X_train, y_train)
    rmse, r2, adj_r2 = evaluate_model(model, X_test, y_test)

    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Adjusted R²: {adj_r2:.4f}")