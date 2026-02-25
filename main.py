"""
Main execution script for Life Expectancy ML Project.

Runs:
1. Data acquisition (Eurostat + World Bank)
2. Data processing & integration
"""

from src.data_fetcher import run_data_acquisition
from src.data_loader import integrate_datasets


def main():

    print("=" * 100)
    print("STEP 1: Data Acquisition")
    print("=" * 100)

    run_data_acquisition()

    print("\n" + "=" * 100)
    print("STEP 2: Data Processing & Integration")
    print("=" * 100)

    try:
        df_master = integrate_datasets()
    except FileNotFoundError as e:
        print("⚠ Some datasets missing. Skipping integration.")
        print(e)
        return

    print("\nMaster dataset created successfully.")
    print("Final dataset shape:", df_master.shape)


if __name__ == "__main__":
    main()