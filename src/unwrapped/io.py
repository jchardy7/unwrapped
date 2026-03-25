"""Utilities for loading the Spotify dataset."""

import pandas as pd

DEFAULT_DATA_PATH = "data/raw/spotify_data.csv"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    return df


def run_validation(path: str):
    from .validation import run_validation as _run_validation

    return _run_validation(path)
