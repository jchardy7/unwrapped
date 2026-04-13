"""Tests for the loading utilities in :mod:`unwrapped.io`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from unwrapped.io import load_data


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Write a minimal Spotify-like CSV and return its path."""
    path = tmp_path / "tracks.csv"
    path.write_text(
        "track_id,artists,track_name,popularity\n"
        "id-1,Artist A,Song A,80\n"
        "id-2,Artist B,Song B,60\n"
    )
    return path


@pytest.fixture
def csv_with_index(tmp_path: Path) -> Path:
    """Write a CSV that includes the common ``Unnamed: 0`` export column."""
    path = tmp_path / "tracks_indexed.csv"
    path.write_text(
        "Unnamed: 0,track_id,artists,track_name,popularity\n"
        "0,id-1,Artist A,Song A,80\n"
        "1,id-2,Artist B,Song B,60\n"
    )
    return path


def test_load_data_reads_csv(sample_csv: Path) -> None:
    """load_data should return a DataFrame with the expected rows and columns."""
    df = load_data(str(sample_csv))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ["track_id", "artists", "track_name", "popularity"]


def test_load_data_drops_unnamed_index_column(csv_with_index: Path) -> None:
    """The ``Unnamed: 0`` column should be silently removed."""
    df = load_data(str(csv_with_index))

    assert "Unnamed: 0" not in df.columns
    assert len(df.columns) == 4


def test_load_data_preserves_columns_without_index(sample_csv: Path) -> None:
    """When no index column is present, all original columns should survive."""
    df = load_data(str(sample_csv))

    assert list(df.columns) == ["track_id", "artists", "track_name", "popularity"]


def test_load_data_raises_for_missing_file() -> None:
    """A nonexistent path should raise an error."""
    with pytest.raises((FileNotFoundError, OSError)):
        load_data("nonexistent/path.csv")
