import numpy as np
from sklearn.model_selection import cross_val_score

from src.ml.preprocessing import (
    clean_data,
    split_data,
    remove_outliers_iqr,
    preprocess_training_data,
    preprocess_test_data,
)
from src.ml.model import train_model
from src.ml.evaluation import evaluate_model, adjusted_r2


def run_ml_pipeline(df):

    print("=" * 60)
    print("ML PIPELINE STARTED")
    print("=" * 60)

    df = clean_data(df, target="life_expectancy")

    # Split BEFORE preprocessing
    X_train, X_test, y_train, y_test = split_data(
        df, target="life_expectancy"
    )

    # Remove outliers from training only
    X_train, y_train = remove_outliers_iqr(X_train, y_train)

    # Preprocess
    X_train_processed, imputer, scaler = preprocess_training_data(X_train)
    X_test_processed = preprocess_test_data(X_test, imputer, scaler)

    # Train
    model = train_model(X_train_processed, y_train)

    # Evaluate
    train_rmse, train_r2, _ = evaluate_model(
        model, X_train_processed, y_train
    )
    test_rmse, test_r2, y_pred = evaluate_model(
        model, X_test_processed, y_test
    )

    adj = adjusted_r2(
        test_r2,
        len(y_test),
        X_test_processed.shape[1]
    )

    # Cross-validation
    cv_scores = cross_val_score(
        model,
        X_train_processed,
        y_train,
        cv=5,
        scoring="r2"
    )

    # PRINT FULL REPORT
    print("\nMODEL REPORT")
    print("-" * 60)
    print(f"Observations (train): {len(y_train)}")
    print(f"Features: {X_train_processed.shape[1]}")

    print("\nTRAIN PERFORMANCE")
    print(f"R²: {train_r2:.4f}")
    print(f"RMSE: {train_rmse:.4f}")

    print("\nTEST PERFORMANCE")
    print(f"R²: {test_r2:.4f}")
    print(f"Adjusted R²: {adj:.4f}")
    print(f"RMSE: {test_rmse:.4f}")

    print("\nCROSS-VALIDATION (5-fold)")
    print(f"Mean R²: {cv_scores.mean():.4f}")
    print(f"Std: {cv_scores.std():.4f}")

    print("\nCOEFFICIENTS")
    for feature, coef in zip(X_train.columns, model.coef_):
        print(f"{feature:<30} {coef:>10.4f}")

    print("=" * 60)
    print("ML FINISHED")
    print("=" * 60)