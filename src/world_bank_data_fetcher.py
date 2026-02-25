"""
Module for fetching indicator data from the World Bank API.
"""

from pathlib import Path
from typing import Dict, Any, List

import requests
import pandas as pd
from pathlib import Path

# List of EU ISO3 country codes
# These codes will be joined using ";" in the API request
EU_ISO3 = [
    "AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA",
    "DEU","GRC","HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD",
    "POL","PRT","ROU","SVK","SVN","ESP","SWE"
]


def fetch_worldbank_indicator(indicator_code: str, filename: str) -> pd.DataFrame:
    
    """
    Fetch data for a given World Bank indicator and save it to CSV.

    Parameters
    ----------
    indicator_code : str
        World Bank indicator code (e.g., 'SP.DYN.TFRT.IN')
    filename : str
        Name of output CSV file (without extension)

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        country, iso3, year, value
    """

    # Join EU country codes into semicolon-separated string
    # Required format for World Bank API
    countries = ";".join(EU_ISO3)

    # Construct API URL
    url = f"https://api.worldbank.org/v2/country/{countries}/indicator/{indicator_code}"

    # Request parameters:
    # - format=json → return JSON format
    # - per_page=20000 → retrieve all results in one request
    params = {
        "format": "json",
        "per_page": 20000
    }

    # Add browser-like header to prevent request blocking
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    #print(f"Requesting indicator: {indicator_code}")
    #print("URL:", url)

    # Make API request
    # Timeout set to 120 seconds because API response is slow
    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=120
    )

    # Raise exception if HTTP request failed (e.g., 500, 502 errors)
    response.raise_for_status()

    # Parse JSON response
    # World Bank returns:
    # data[0] → metadata
    # data[1] → actual records
    data = response.json()

    # Ensure response format is valid
    if len(data) < 2:
        raise ValueError("Unexpected API response format")

    records = data[1]

    # Convert JSON records to pandas DataFrame
    df = pd.DataFrame(records)

    # Clean and restructure dataset

    # Extract country name from nested dictionary
    df["countryName"] = df["country"].apply(lambda x: x["value"])

    # Create ISO3 column
    df["country"] = df["countryiso3code"]

    # Rename date column to year
    df["year"] = df["date"]

    # Keep only relevant columns
    df = df[["countryName", "country", "year", "value"]]

    # Remove rows with missing indicator values
    df = df.dropna(subset=["value"])

    # Convert year column to integer
    df["year"] = df["year"].astype(int)


    # Save cleaned dataset to data/raw directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_dir / f"{filename}.csv", index=False)

    #print(f"{filename}.csv saved successfully.")

    return df

    # Individual Indicator Wrappers

def fetch_gdp_per_capita() -> pd.DataFrame:
    """Fetch GDP per capita (current US$)."""
    return fetch_worldbank_indicator("NY.GDP.PCAP.CD", "gdp_per_capita")


def fetch_urban_population_percentage() -> pd.DataFrame:
    """Fetch urban population (% of total population)."""
    return fetch_worldbank_indicator("SP.URB.TOTL.IN.ZS", "urban_population_pct")


def fetch_fertility_rate() -> pd.DataFrame:
    """Fetch fertility rate (births per woman)."""
    return fetch_worldbank_indicator("SP.DYN.TFRT.IN", "fertility_rate")


def fetch_population_density() -> pd.DataFrame:
    """Fetch population density (people per sq. km)."""
    return fetch_worldbank_indicator("EN.POP.DNST", "population_density")