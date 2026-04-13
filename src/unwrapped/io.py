"""Utilities for loading the Spotify dataset."""

import pandas as pd

DEFAULT_DATA_PATH = "data/raw/spotify_data.csv"


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

    # Some CSV exports include the old row index as an extra column.
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df
