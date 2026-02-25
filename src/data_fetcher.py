"""
Data acquisition orchestration module.

Coordinates fetching data from multiple sources (Eurostat and World Bank)
with error handling and logging.
"""

import logging
from typing import Callable, List

from src.eurostat_data_fetcher import (
    fetch_doctors_per_100k,
    fetch_hospital_capacity,
    fetch_household_expenditure,
    fetch_life_expectancy,
    fetch_gov_health_expenditure,
)
from src.world_bank_data_fetcher import (
    fetch_fertility_rate,
    fetch_gdp_per_capita,
    fetch_population_density,
    fetch_urban_population_percentage,
)

# Configure logger
logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Set up logging configuration for the data acquisition module."""
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)


# Define lists of fetcher functions by source
EUROSTAT_FETCHERS: List[Callable[[], None]] = [
    fetch_life_expectancy,
    fetch_doctors_per_100k,
    fetch_household_expenditure,
    fetch_hospital_capacity,
    fetch_gov_health_expenditure,
]

WORLD_BANK_FETCHERS: List[Callable[[], None]] = [
    fetch_gdp_per_capita,
    fetch_urban_population_percentage,
    fetch_fertility_rate,
    fetch_population_density,
]


def run_data_acquisition() -> None:
    """
    Orchestrate data acquisition from all sources.

    Fetches data from Eurostat and World Bank APIs, with per-dataset error
    handling to ensure that failures in one dataset do not halt the entire
    data acquisition pipeline.

    Logs progress and results for each dataset fetch and a summary at the end.
    """
    _setup_logging()

    logger.info("=" * 60)
    logger.info("Starting data acquisition from Eurostat")
    logger.info("=" * 60)

    # Track results
    successful_fetches = []
    failed_fetches = []

    # Fetch Eurostat data

    for fetcher in EUROSTAT_FETCHERS:
        dataset_name = fetcher.__name__
        try:
            logger.info(f"Fetching {dataset_name}...")
            fetcher()
            logger.info(f"✓ {dataset_name} completed successfully")
            successful_fetches.append(dataset_name)
        except Exception as e:
            logger.error(
                f"✗ {dataset_name} failed with error: {type(e).__name__}: {str(e)}"
            )
            failed_fetches.append((dataset_name, str(e)))

    logger.info("=" * 60)
    logger.info("Starting data acquisition from World Bank")
    logger.info("=" * 60)

    # Fetch World Bank data

    for fetcher in WORLD_BANK_FETCHERS:
        dataset_name = fetcher.__name__
        try:
            logger.info(f"Fetching {dataset_name}...")
            fetcher()
            logger.info(f"✓ {dataset_name} completed successfully")
            successful_fetches.append(dataset_name)
        except Exception as e:
            logger.error(
                f"✗ {dataset_name} failed with error: {type(e).__name__}: {str(e)}"
            )
            failed_fetches.append((dataset_name, str(e)))

    # Log summary
    total_datasets = len(successful_fetches) + len(failed_fetches)
    logger.info("=" * 60)
    logger.info("Data acquisition completed")
    logger.info("=" * 60)
    logger.info(
        f"Summary: {len(successful_fetches)}/{total_datasets} datasets "
        f"fetched successfully"
    )

    if successful_fetches:
        logger.info("Successful datasets:")
        for name in successful_fetches:
            logger.info(f"  ✓ {name}")

    if failed_fetches:
        logger.warning("Failed datasets:")
        for name, error in failed_fetches:
            logger.warning(f"  ✗ {name}: {error}")

    logger.info("=" * 60)
