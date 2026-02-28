"""
Main Execution Script
---------------------

This script orchestrates the full Life Expectancy ML pipeline.

Pipeline Steps:
1. Data Acquisition (Eurostat + World Bank APIs)
2. Data Processing & Integration
3. Exploratory Data Summary
4. Visualisation Generation

This file should be executed from the project root directory.
"""

# =============================
# Import Pipeline Components
# =============================

from src.data_fetcher import run_data_acquisition
from src.data_loader import integrate_datasets
from src.exploratory_data_analysis import run_eda_summary
from src.visualizations import run_visualisations
from src.ml.preprocessing import clean_data, handle_missing_values, remove_outliers_iqr, split_data
from src.ml.model import train_model
from src.ml.evaluation import evaluate_model

def main() -> None:
    """
    Execute the full end-to-end data pipeline.
    """

    # STEP 1: Data Acquisition
    print("=" * 100)
    print("STEP 1: Data Acquisition")
    print("=" * 100)

    run_data_acquisition()

    # STEP 2: Data Integration
    print("\n" + "=" * 100)
    print("STEP 2: Data Processing & Integration")
    print("=" * 100)

    try:
        df_master = integrate_datasets()
    except FileNotFoundError as e:
        print("⚠ Some datasets missing. Skipping integration.")
        print(e)
        return

    print("\n✓ Master dataset created successfully.")
    print("Final dataset shape:", df_master.shape)

    # STEP 3: Exploratory Data Summary
    print("\n" + "=" * 100)
    print("STEP 3: Exploratory Data Summary")
    print("=" * 100)

    run_eda_summary()

    # STEP 4: Generating Visualisations
    print("\n" + "=" * 100)
    print("STEP 4: Generating Visualisations")
    print("=" * 100)

    try:
        run_visualisations(df_master)
    except Exception as e:
        print("⚠ Error during visualisation execution.")
        print(e)
        return

    print("\n✓ Full pipeline completed successfully!")

    # STEP 5: ML Part - Data Preprocessing, Model Training, Evaluation
    print("\n" + "=" * 100)
    print("STEP 5: Machine Learning Model")
    print("=" * 100)

    try:
        df_ml = clean_data(df_master)
        df_ml = handle_missing_values(df_ml, method="mean")
        df_ml = remove_outliers_iqr(df_ml, target="life_expectancy")

        X_train, X_test, y_train, y_test = split_data(df_ml, target="life_expectancy")

        model = train_model(X_train, y_train)

        rmse, r2, adj_r2 = evaluate_model(model, X_test, y_test)

        print(f"\n✓ Model trained successfully")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"Adjusted R²: {adj_r2:.4f}")

    except Exception as e:
        print("⚠ Error during ML pipeline execution.")
        print(e)
        return

# Entry point of the script
if __name__ == "__main__":
    main()