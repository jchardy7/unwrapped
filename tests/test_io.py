"""Tests for the loading and IO-adjacent entrypoints."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from unwrapped.clean import run_cleaning
from unwrapped.io import load_data
from unwrapped.validation import run_validation


def make_valid_row(**overrides: Any) -> dict[str, Any]:
    """Build a single row that satisfies the cleaning and validation rules."""

    row = {
        "track_id": "track-1",
        "artists": "Artist 1",
        "album_name": "Album 1",
        "track_name": "Song 1",
        "popularity": 55,
        "duration_ms": 180000,
        "explicit": False,
        "danceability": 0.5,
        "energy": 0.8,
        "key": 5,
        "loudness": -5.0,
        "mode": 1,
        "speechiness": 0.2,
        "acousticness": 0.3,
        "instrumentalness": 0.1,
        "liveness": 0.15,
        "valence": 0.6,
        "tempo": 120.0,
        "time_signature": 4,
        "track_genre": "pop",
    }
    row.update(overrides)
    return row


def make_valid_df() -> pd.DataFrame:
    """Return a compact valid DataFrame for IO entrypoint tests."""

    return pd.DataFrame(
        [
            make_valid_row(track_id="track-1", popularity=40, loudness=-6.0),
            make_valid_row(
                track_id="track-2",
                track_name="Song 2",
                popularity=70,
                energy=0.85,
                loudness=-5.0,
            ),
            make_valid_row(
                track_id="track-3",
                track_name="Song 3",
                popularity=85,
                energy=0.9,
                loudness=-4.0,
            ),
        ]
    )


def test_load_data_reads_csv_and_drops_export_index_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`load_data` should delegate to pandas and strip `Unnamed: 0`."""

    raw_df = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "track_id": ["id-1", "id-2"],
            "artists": ["Artist A", "Artist B"],
            "track_name": ["Song A", "Song B"],
            "popularity": [80, 60],
        }
    )
    calls: list[str] = []

    def fake_read_csv(path: str) -> pd.DataFrame:
        calls.append(path)
        return raw_df

    monkeypatch.setattr("unwrapped.io.pd.read_csv", fake_read_csv)

    df = load_data("fake/path.csv")

    assert calls == ["fake/path.csv"]
    assert list(df.columns) == ["track_id", "artists", "track_name", "popularity"]
    assert df.to_dict("records") == [
        {
            "track_id": "id-1",
            "artists": "Artist A",
            "track_name": "Song A",
            "popularity": 80,
        },
        {
            "track_id": "id-2",
            "artists": "Artist B",
            "track_name": "Song B",
            "popularity": 60,
        },
    ]


def test_load_data_preserves_columns_when_no_index_column_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`load_data` should return the original columns when no export index exists."""

    raw_df = pd.DataFrame(
        {
            "track_id": ["id-1"],
            "artists": ["Artist A"],
            "track_name": ["Song A"],
            "popularity": [80],
        }
    )

    monkeypatch.setattr("unwrapped.io.pd.read_csv", lambda _: raw_df)

    df = load_data("fake/path.csv")

    assert list(df.columns) == ["track_id", "artists", "track_name", "popularity"]
    assert df.equals(raw_df)


def test_load_data_propagates_read_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """File-loading errors from pandas should surface to callers."""

    def fake_read_csv(path: str) -> pd.DataFrame:
        raise FileNotFoundError(path)

    monkeypatch.setattr("unwrapped.io.pd.read_csv", fake_read_csv)

    with pytest.raises(FileNotFoundError, match="missing.csv"):
        load_data("missing.csv")


def test_run_cleaning_loads_raw_data_and_returns_cleaned_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run_cleaning` should load once, clean the data, and return the report."""

    raw_df = pd.DataFrame(
        [
            make_valid_row(
                track_id="track-1",
                artists="  Artist 1  ",
                explicit="TRUE",
                popularity=60,
            ),
            make_valid_row(
                track_id="track-1",
                artists="Artist 1",
                popularity=90,
            ),
            make_valid_row(
                track_id="track-2",
                track_name="   ",
                explicit="maybe",
            ),
        ]
    ).assign(**{"Unnamed: 0": [0, 1, 2]})
    calls: list[str] = []

    def fake_load_data(path: str) -> pd.DataFrame:
        calls.append(path)
        return raw_df

    monkeypatch.setattr("unwrapped.io.load_data", fake_load_data)

    cleaned, report = run_cleaning("fake/path.csv")

    assert calls == ["fake/path.csv"]
    assert list(cleaned["track_id"]) == ["track-1"]
    assert cleaned.loc[0, "artists"] == "Artist 1"
    assert cleaned.loc[0, "explicit"] == False
    assert report["index_columns_removed"] == 1
    assert report["blank_text_values"]["track_name"] == 1
    assert report["rows_removed_total"] == 2


def test_run_validation_loads_data_and_returns_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run_validation` should validate the loaded data and return its report."""

    df = make_valid_df()
    calls: list[str] = []

    def fake_load_data(path: str) -> pd.DataFrame:
        calls.append(path)
        return df

    monkeypatch.setattr("unwrapped.io.load_data", fake_load_data)

    returned_df, report = run_validation("fake/path.csv")

    assert calls == ["fake/path.csv"]
    assert returned_df.equals(df)
    assert report["num_rows"] == len(df)
    assert report["num_columns"] == len(df.columns)
    assert report["unique_tracks"] == 3


def test_run_validation_raises_when_loaded_data_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid loaded data should fail through the public validation entrypoint."""

    invalid_df = make_valid_df().drop(columns=["tempo"])

    monkeypatch.setattr("unwrapped.io.load_data", lambda _: invalid_df)

    with pytest.raises(ValueError, match="Missing columns"):
        run_validation("fake/path.csv")
