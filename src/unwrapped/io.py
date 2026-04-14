"""Utilities for loading the Spotify dataset."""

import pandas as pd

DEFAULT_DATA_PATH = "data/raw/spotify_data.csv"

def _drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the Unnamed: 0 export index column if present."""
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

def load_data(path: str) -> pd.DataFrame:
    """Load the raw Spotify CSV into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
        Raw Spotify dataset with the common export index column removed when
        present.
    """

    df = pd.read_csv(path)
    return _drop_index_column(df)

def load_json(path: str) -> pd.DataFrame:
    """Load a Spotify JSON export into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the JSON file to load.

    Returns
    -------
    pd.DataFrame
        Raw Spotify dataset with the common export index column removed when
        present.
    """
    df = pd.read_json(path)
    return _drop_index_column(df)

