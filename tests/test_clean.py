"""Tests for the cleaning helpers in :mod:`unwrapped.clean`.

These tests use small, explicit DataFrames so each cleaning rule can be
verified independently and failures point back to a single behavior.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from unwrapped.clean import clean_data
from unwrapped.clean import coerce_numeric_columns
from unwrapped.clean import deduplicate_tracks
from unwrapped.clean import drop_index_column
from unwrapped.clean import handle_missing_values
from unwrapped.clean import normalize_explicit_column
from unwrapped.clean import remove_invalid_rows
from unwrapped.clean import run_cleaning
from unwrapped.clean import standardize_text_fields
from unwrapped.clean import validate_cleaning_columns


def make_valid_row(**overrides: Any) -> dict[str, Any]:
    """Build a single row that satisfies the current cleaning rules."""

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
    """Return a minimal valid DataFrame for use as a test starting point."""

    return pd.DataFrame(
        [
            make_valid_row(track_id="track-1", popularity=40),
            make_valid_row(track_id="track-2", popularity=70, track_name="Song 2"),
            make_valid_row(track_id="track-3", popularity=85, track_name="Song 3"),
        ]
    )


def test_validate_cleaning_columns_raises_for_missing_columns() -> None:
    """Cleaning should fail fast when required project columns are missing."""

    df = make_valid_df().drop(columns=["tempo", "track_name"])

    with pytest.raises(ValueError) as exc_info:
        validate_cleaning_columns(df)

    message = str(exc_info.value)
    assert "missing required columns" in message.lower()
    assert "tempo" in message
    assert "track_name" in message


def test_drop_index_column_removes_export_column_when_present() -> None:
    """The export index column should be dropped and reported."""

    df = make_valid_df().assign(**{"Unnamed: 0": [0, 1, 2]})

    cleaned, dropped = drop_index_column(df)

    assert "Unnamed: 0" not in cleaned.columns
    assert dropped == 1


def test_standardize_text_fields_trims_text_and_marks_blanks_missing() -> None:
    """Whitespace-only strings should become missing after trimming."""

    df = make_valid_df()
    df.loc[0, "artists"] = "  Artist 1  "
    df.loc[1, "album_name"] = "   "

    cleaned, blank_counts = standardize_text_fields(df)

    assert cleaned.loc[0, "artists"] == "Artist 1"
    assert pd.isna(cleaned.loc[1, "album_name"])
    assert blank_counts["album_name"] == 1


def test_normalize_explicit_column_maps_known_values_and_coerces_unknowns() -> None:
    """Mixed boolean encodings should normalize to nullable booleans."""

    df = make_valid_df()
    df["explicit"] = ["TRUE", "no", "maybe"]

    cleaned, coerced_missing = normalize_explicit_column(df)

    assert cleaned["explicit"].tolist() == [True, False, pd.NA]
    assert str(cleaned["explicit"].dtype) == "boolean"
    assert coerced_missing == 1


def test_coerce_numeric_columns_counts_values_converted_to_missing() -> None:
    """Non-numeric strings should become missing in numeric columns."""

    df = make_valid_df()
    df["tempo"] = df["tempo"].astype(object)
    df["popularity"] = df["popularity"].astype(object)
    df.loc[0, "tempo"] = "fast"
    df.loc[1, "popularity"] = "very popular"

    cleaned, coercion_counts = coerce_numeric_columns(df)

    assert pd.isna(cleaned.loc[0, "tempo"])
    assert pd.isna(cleaned.loc[1, "popularity"])
    assert coercion_counts["tempo"] == 1
    assert coercion_counts["popularity"] == 1


def test_handle_missing_values_drops_rows_missing_required_text_fields() -> None:
    """Rows missing core identifiers should be removed before analysis."""

    df = make_valid_df()
    df.loc[1, "track_name"] = pd.NA
    df.loc[2, "artists"] = pd.NA

    cleaned, missing_counts = handle_missing_values(df)

    assert len(cleaned) == 1
    assert missing_counts["track_name"] == 1
    assert missing_counts["artists"] == 1


def test_remove_invalid_rows_marks_invalid_numeric_values_and_drops_rows() -> None:
    """Rows with impossible core numeric values should be removed."""

    df = make_valid_df()
    df.loc[0, "danceability"] = 1.5
    df.loc[1, "tempo"] = 0
    df.loc[2, "duration_ms"] = -10

    cleaned, invalid_counts = remove_invalid_rows(df)

    assert cleaned.empty
    assert invalid_counts["danceability"] == 1
    assert invalid_counts["tempo"] == 1
    assert invalid_counts["duration_ms"] == 1


def test_deduplicate_tracks_keeps_most_complete_then_most_popular_row() -> None:
    """Repeated track IDs should keep the best canonical row."""

    df = pd.DataFrame(
        [
            make_valid_row(track_id="track-1", popularity=10, album_name=""),
            make_valid_row(track_id="track-1", popularity=50, album_name="Album A"),
            make_valid_row(track_id="track-2", track_name="Song 2"),
        ]
    )

    cleaned, summary = deduplicate_tracks(df)

    kept_row = cleaned.loc[cleaned["track_id"] == "track-1"].iloc[0]
    assert len(cleaned) == 2
    assert kept_row["popularity"] == 50
    assert kept_row["album_name"] == "Album A"
    assert summary["duplicate_track_ids_removed"] == 1


def test_clean_data_runs_full_pipeline_and_reports_changes() -> None:
    """The cleaning entrypoint should combine all cleaning stages."""

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
                album_name="Album 1",
                popularity=90,
            ),
            make_valid_row(
                track_id="track-2",
                track_name="   ",
                explicit="maybe",
            ),
            make_valid_row(
                track_id="track-3",
                danceability=2.0,
            ),
        ]
    ).assign(**{"Unnamed: 0": [0, 1, 2, 3]})

    cleaned, report = clean_data(raw_df)

    assert list(cleaned["track_id"]) == ["track-1"]
    assert cleaned.loc[0, "artists"] == "Artist 1"
    assert cleaned.loc[0, "explicit"] == False
    assert report["index_columns_removed"] == 1
    assert report["blank_text_values"]["track_name"] == 1
    assert report["explicit_values_coerced_to_missing"] == 1
    assert report["invalid_values_detected"]["danceability"] == 1
    assert report["duplicate_track_ids_removed"] == 1
    assert report["rows_removed_total"] == 3


def test_run_cleaning_loads_data_once_and_returns_cleaned_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`run_cleaning` should delegate loading to the IO layer once."""

    df = make_valid_df()
    calls: list[str] = []

    def fake_load_data(path: str) -> pd.DataFrame:
        """Record the path so the entrypoint behavior can be asserted."""

        calls.append(path)
        return df

    monkeypatch.setattr("unwrapped.io.load_data", fake_load_data)

    cleaned, report = run_cleaning("fake/path.csv")

    assert calls == ["fake/path.csv"]
    assert len(cleaned) == len(df)
    assert list(cleaned.columns) == list(df.columns)
    assert list(cleaned["track_id"]) == list(df["track_id"])
    assert report["input_rows"] == len(df)
    assert report["rows_removed_total"] == 0
