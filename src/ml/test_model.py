import numpy as np
import pandas as pd

from src.ml.preprocessing import (
    clean_data,
    split_data,
    remove_outliers_iqr,
    preprocess_training_data,
    preprocess_test_data,
)

from src.ml.model import train_model
from src.ml.evaluation import evaluate_model


def test_full_pipeline():
    """
    Tests full ML pipeline using synthetic regression data.
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

    df = clean_data(df, target="life_expectancy")

    X_train, X_test, y_train, y_test = split_data(
        df, target="life_expectancy"
    )

    X_train, y_train = remove_outliers_iqr(X_train, y_train)

    X_train_processed, imputer, scaler = preprocess_training_data(X_train)
    X_test_processed = preprocess_test_data(X_test, imputer, scaler)

    model = train_model(X_train_processed, y_train)

    _, r2, _ = evaluate_model(model, X_test_processed, y_test)

    assert r2 > 0.7