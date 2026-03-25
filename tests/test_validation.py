"""Tests for the validation helpers in :mod:`unwrapped.validation`.

These tests focus on small, explicit DataFrame inputs so each validation rule
is easy to reason about and failures point back to a single behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

# Add the local src/ directory so the package can be imported without requiring
# an editable install before running the tests in a fresh checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unwrapped.validation import EXPECTED_COLUMNS
from unwrapped.validation import missing_summary
from unwrapped.validation import run_validation
from unwrapped.validation import validate_correlations
from unwrapped.validation import validate_duplicates
from unwrapped.validation import validate_ranges
from unwrapped.validation import validate_schema
from unwrapped.validation import validate_track_consistency
from unwrapped.validation import validation_report


def make_valid_row(**overrides: Any) -> dict[str, Any]:
    """Build a single row that satisfies the current validation rules.

    A shared row factory keeps each test compact while still letting us tweak
    only the field that matters for the edge case under test.
    """

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
            make_valid_row(track_id="track-1", loudness=-6.0, energy=0.8),
            make_valid_row(track_id="track-2", loudness=-5.0, energy=0.85),
            make_valid_row(track_id="track-3", loudness=-4.0, energy=0.9),
        ]
    )


def test_validate_schema_allows_expected_columns_only() -> None:
    """`validate_schema` should pass silently for the exact expected schema."""

    df = make_valid_df()[sorted(EXPECTED_COLUMNS)]

    validate_schema(df)


def test_validate_schema_raises_for_missing_columns() -> None:
    """Missing required columns should fail fast with a helpful error."""

    df = make_valid_df().drop(columns=["track_name", "tempo"])

    with pytest.raises(ValueError) as exc_info:
        validate_schema(df)

    message = str(exc_info.value)
    assert "Missing columns" in message
    # Checking the actual missing field names makes the assertion more precise
    # without coupling the test to the set ordering in the exception message.
    assert "track_name" in message
    assert "tempo" in message


def test_validate_schema_warns_when_extra_columns_are_present(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unexpected columns should warn without blocking the rest of the flow."""

    df = make_valid_df().assign(source="demo")

    validate_schema(df)

    captured = capsys.readouterr()
    assert "Extra columns detected" in captured.out
    assert "source" in captured.out


def test_validate_ranges_accepts_boundary_values() -> None:
    """Inclusive boundaries like 0 and 1 should count as valid."""

    df = pd.DataFrame(
        [
            make_valid_row(
                danceability=0.0,
                energy=1.0,
                valence=0.0,
                acousticness=1.0,
                instrumentalness=0.0,
                speechiness=1.0,
                tempo=1.0,
                duration_ms=1,
            )
        ]
    )

    validate_ranges(df)


def test_validate_ranges_aggregates_multiple_failures() -> None:
    """Multiple invalid fields should be reported in the same exception."""

    df = pd.DataFrame(
        [
            make_valid_row(
                danceability=1.5,
                energy=-0.1,
                tempo=0,
                duration_ms=-10,
            )
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        validate_ranges(df)

    message = str(exc_info.value)
    # We assert each fragment separately so the test stays stable even if the
    # ordering of checks changes in the implementation later.
    assert "danceability out of range [0, 1]" in message
    assert "energy out of range [0, 1]" in message
    assert "tempo contains non-positive values" in message
    assert "duration_ms contains non-positive values" in message


def test_validate_ranges_treats_missing_numeric_values_as_invalid() -> None:
    """NaN in checked range columns should fail instead of silently passing."""

    df = pd.DataFrame([make_valid_row(energy=float("nan"))])

    with pytest.raises(ValueError) as exc_info:
        validate_ranges(df)

    assert "energy out of range [0, 1]" in str(exc_info.value)


def test_validate_duplicates_counts_both_row_and_track_id_duplicates() -> None:
    """Duplicate rows and duplicate track IDs should be tracked separately."""

    duplicated_row = make_valid_row(track_id="track-1", track_name="Song 1")
    unique_same_track = make_valid_row(track_id="track-1", track_name="Remix")
    df = pd.DataFrame([duplicated_row, duplicated_row, unique_same_track])

    result = validate_duplicates(df)

    assert result == {"duplicate_rows": 1, "duplicate_track_ids": 2}


def test_validate_track_consistency_counts_tracks_with_mixed_core_metrics() -> None:
    """A reused track ID with conflicting metrics should count as inconsistent."""

    df = pd.DataFrame(
        [
            make_valid_row(track_id="track-1", danceability=0.4),
            make_valid_row(track_id="track-1", danceability=0.9),
            # This second track is duplicated consistently and should not count.
            make_valid_row(track_id="track-2", track_name="Song 2"),
            make_valid_row(track_id="track-2", track_name="Song 2"),
        ]
    )

    assert validate_track_consistency(df) == 1


def test_validate_correlations_warns_for_weak_relationship(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Weak energy/loudness correlation should surface as a warning."""

    df = make_valid_df().copy()
    # These values are intentionally shaped to break the expected positive
    # relationship between energy and loudness.
    df["energy"] = [0.1, 0.9, 0.2]
    df["loudness"] = [-3.0, -6.0, -4.0]

    validate_correlations(df)

    captured = capsys.readouterr()
    assert "Weak correlation between energy and loudness" in captured.out


def test_validate_correlations_stays_quiet_for_strong_relationship(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Strong positive correlation should not emit a warning."""

    validate_correlations(make_valid_df())

    captured = capsys.readouterr()
    assert captured.out == ""


def test_missing_summary_returns_null_counts_per_column() -> None:
    """`missing_summary` should expose raw per-column null counts."""

    df = make_valid_df().copy()
    df.loc[0, "artists"] = None
    df.loc[1, "tempo"] = None

    result = missing_summary(df)

    assert result["artists"] == 1
    assert result["tempo"] == 1
    assert result["track_id"] == 0


def test_validation_report_combines_summary_metrics() -> None:
    """The report should combine missing, duplicate, uniqueness, and consistency data."""

    df = pd.DataFrame(
        [
            make_valid_row(track_id="track-1", artists=None),
            make_valid_row(track_id="track-1"),
            make_valid_row(track_id="track-2", track_name="Song 2"),
        ]
    )

    report = validation_report(df)

    assert report["num_rows"] == 3
    assert report["num_columns"] == len(EXPECTED_COLUMNS)
    assert report["missing_values"]["artists"] == 1
    assert report["duplicate_rows"] == 0
    assert report["duplicate_track_ids"] == 1
    assert report["unique_tracks"] == 2
    assert report["inconsistent_tracks"] == 0


def test_run_validation_loads_data_and_returns_report(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run_validation` should load data once and return the validated report."""

    df = make_valid_df()
    calls: list[str] = []

    def fake_load_data(path: str) -> pd.DataFrame:
        """Record the path so the integration flow can be asserted."""

        calls.append(path)
        return df

    # Patching the IO module keeps this test independent from the real CSV file
    # while still exercising the validation entrypoint end to end.
    monkeypatch.setattr("unwrapped.io.load_data", fake_load_data)

    returned_df, report = run_validation("fake/path.csv")

    assert calls == ["fake/path.csv"]
    assert returned_df.equals(df)
    assert report["num_rows"] == len(df)
    assert report["unique_tracks"] == df["track_id"].nunique()


def test_run_validation_propagates_validation_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry-point validation should fail loudly when loaded data is invalid."""

    invalid_df = make_valid_df().drop(columns=["tempo"])

    def fake_load_data(path: str) -> pd.DataFrame:
        """Return an invalid frame so the schema check fails immediately."""

        return invalid_df

    monkeypatch.setattr("unwrapped.io.load_data", fake_load_data)

    with pytest.raises(ValueError) as exc_info:
        run_validation("fake/path.csv")

    assert "Missing columns" in str(exc_info.value)
