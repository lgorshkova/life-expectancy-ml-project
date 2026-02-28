import pandas as pd
from src.data_loader import integrate_datasets
from src.ml.preprocessing import clean_data, handle_missing_values, remove_outliers_iqr, split_data
from src.ml.model import train_model
from src.ml.evaluation import evaluate_model

def main():
    """
    Runs the ML pipeline for life_expectancy prediction.
    """
    df = integrate_datasets()
    df = clean_data(df)
    df = handle_missing_values(df, method="mean")
    df = remove_outliers_iqr(df, target="life_expectancy")
    X_train, X_test, y_train, y_test = split_data(df, target="life_expectancy")
    model = train_model(X_train, y_train)
    rmse, r2, adj_r2 = evaluate_model(model, X_test, y_test)
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Adjusted R²: {adj_r2:.4f}")

if __name__ == "__main__":
    main()