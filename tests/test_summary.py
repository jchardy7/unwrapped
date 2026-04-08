"""Tests for the descriptive summary helpers in :mod:`unwrapped.summary`.

These tests use small, explicit DataFrames so each summary function can be
verified independently and failures point back to a single behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unwrapped.summary import (
    correlation_matrix,
    describe_categorical,
    describe_missing,
    describe_numeric,
    describe_shape,
    detect_outliers,
    genre_summary,
    run_summary,
    summarize_data,
    target_correlations,
)


def make_valid_row(**overrides: Any) -> dict[str, Any]:
    """Build a single row that satisfies the dataset schema."""

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


def make_test_df() -> pd.DataFrame:
    """Return a small DataFrame with enough variation for summary tests."""

    return pd.DataFrame(
        [
            make_valid_row(
                track_id="track-1",
                popularity=40,
                danceability=0.3,
                energy=0.6,
                track_genre="pop",
                artists="Artist A",
            ),
            make_valid_row(
                track_id="track-2",
                popularity=70,
                danceability=0.7,
                energy=0.9,
                track_genre="rock",
                artists="Artist B",
            ),
            make_valid_row(
                track_id="track-3",
                popularity=85,
                danceability=0.9,
                energy=0.4,
                track_genre="pop",
                artists="Artist A",
            ),
            make_valid_row(
                track_id="track-4",
                popularity=20,
                danceability=0.5,
                energy=0.5,
                track_genre="jazz",
                artists="Artist C",
            ),
        ]
    )


class TestDescribeShape:
    def test_returns_correct_counts(self) -> None:
        df = make_test_df()
        result = describe_shape(df)

        assert result["num_rows"] == 4
        assert result["num_columns"] == len(df.columns)
        assert set(result["columns"]) == set(df.columns)

    def test_dtype_counts_is_populated(self) -> None:
        df = make_test_df()
        result = describe_shape(df)

        assert isinstance(result["dtype_counts"], dict)
        assert sum(result["dtype_counts"].values()) == len(df.columns)


class TestDescribeNumeric:
    def test_returns_expected_stats(self) -> None:
        df = make_test_df()
        result = describe_numeric(df)

        assert "popularity" in result
        stats = result["popularity"]
        assert stats["count"] == 4
        assert stats["min"] == 20.0
        assert stats["max"] == 85.0
        assert "mean" in stats
        assert "std" in stats
        assert "q1" in stats
        assert "median" in stats
        assert "q3" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats

    def test_median_is_correct(self) -> None:
        df = make_test_df()
        result = describe_numeric(df)

        # popularity values: 20, 40, 70, 85 -> median = 55
        assert result["popularity"]["median"] == 55.0

    def test_skips_empty_numeric_column(self) -> None:
        df = make_test_df()
        df["popularity"] = pd.NA
        result = describe_numeric(df)

        assert "popularity" not in result


class TestDescribeCategorical:
    def test_returns_unique_counts_and_top_values(self) -> None:
        df = make_test_df()
        result = describe_categorical(df)

        assert "track_genre" in result
        genre_info = result["track_genre"]
        assert genre_info["num_unique"] == 3
        assert "pop" in genre_info["top_values"]
        assert genre_info["top_values"]["pop"] == 2

    def test_top_n_limits_output(self) -> None:
        df = make_test_df()
        result = describe_categorical(df, top_n=1)

        assert len(result["track_genre"]["top_values"]) == 1

    def test_artists_counted_correctly(self) -> None:
        df = make_test_df()
        result = describe_categorical(df)

        assert result["artists"]["num_unique"] == 3
        assert result["artists"]["top_values"]["Artist A"] == 2


class TestDescribeMissing:
    def test_no_missing_all_zeros(self) -> None:
        df = make_test_df()
        result = describe_missing(df)

        for col_info in result.values():
            assert col_info["count"] == 0
            assert col_info["percentage"] == 0.0

    def test_detects_missing_values(self) -> None:
        df = make_test_df()
        df.loc[0, "popularity"] = pd.NA
        df.loc[1, "popularity"] = pd.NA
        result = describe_missing(df)

        assert result["popularity"]["count"] == 2
        assert result["popularity"]["percentage"] == 50.0


class TestDetectOutliers:
    def test_detects_outlier_beyond_iqr(self) -> None:
        df = make_test_df()
        # Add a row with an extreme popularity value
        outlier_row = make_valid_row(
            track_id="track-5",
            popularity=100,
            danceability=0.5,
            track_genre="pop",
        )
        df = pd.concat([df, pd.DataFrame([outlier_row])], ignore_index=True)

        # Also add a very low one to widen the test
        low_row = make_valid_row(
            track_id="track-6",
            popularity=0,
            danceability=0.5,
            track_genre="pop",
        )
        df = pd.concat([df, pd.DataFrame([low_row])], ignore_index=True)

        result = detect_outliers(df)
        assert "popularity" in result
        assert result["popularity"]["lower_bound"] <= result["popularity"]["upper_bound"]

    def test_returns_bounds(self) -> None:
        df = make_test_df()
        result = detect_outliers(df)

        for col_info in result.values():
            assert "lower_bound" in col_info
            assert "upper_bound" in col_info
            assert "count" in col_info
            assert "percentage" in col_info


class TestCorrelationMatrix:
    def test_returns_symmetric_matrix(self) -> None:
        df = make_test_df()
        corr = correlation_matrix(df)

        assert isinstance(corr, pd.DataFrame)
        assert corr.shape[0] == corr.shape[1]
        # Columns with nonzero variance should have 1.0 on the diagonal;
        # columns with zero variance produce NaN which is expected.
        for col in corr.columns:
            diag = corr.loc[col, col]
            if not pd.isna(diag):
                assert abs(diag - 1.0) < 1e-10

    def test_contains_audio_feature_columns(self) -> None:
        df = make_test_df()
        corr = correlation_matrix(df)

        assert "danceability" in corr.columns
        assert "energy" in corr.columns


class TestTargetCorrelations:
    def test_returns_correlations_sorted_by_abs_value(self) -> None:
        df = make_test_df()
        result = target_correlations(df)

        assert isinstance(result, dict)
        values = list(result.values())
        abs_values = [abs(v) for v in values]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_excludes_target_column_itself(self) -> None:
        df = make_test_df()
        result = target_correlations(df)

        assert "popularity" not in result

    def test_returns_empty_when_target_missing(self) -> None:
        df = make_test_df().drop(columns=["popularity"])
        result = target_correlations(df)

        assert result == {}


class TestGenreSummary:
    def test_returns_per_genre_aggregations(self) -> None:
        df = make_test_df()
        result = genre_summary(df)

        assert isinstance(result, pd.DataFrame)
        assert "pop" in result.index
        assert "rock" in result.index
        assert "jazz" in result.index

    def test_count_matches_genre_frequency(self) -> None:
        df = make_test_df()
        result = genre_summary(df)

        assert result.loc["pop", ("danceability", "count")] == 2
        assert result.loc["rock", ("danceability", "count")] == 1

    def test_returns_empty_when_no_genre_column(self) -> None:
        df = make_test_df().drop(columns=["track_genre"])
        result = genre_summary(df)

        assert result.empty


class TestSummarizeData:
    def test_returns_all_report_sections(self) -> None:
        df = make_test_df()
        report = summarize_data(df)

        expected_keys = [
            "shape",
            "numeric",
            "categorical",
            "missing",
            "outliers",
            "correlation_matrix",
            "target_correlations",
            "genre_summary",
        ]
        for key in expected_keys:
            assert key in report, f"Missing report section: {key}"


class TestRunSummary:
    def test_loads_data_and_returns_summary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`run_summary` should delegate loading to the IO layer."""

        df = make_test_df()
        calls: list[str] = []

        def fake_load_data(path: str) -> pd.DataFrame:
            calls.append(path)
            return df

        monkeypatch.setattr("unwrapped.io.load_data", fake_load_data)

        result_df, report = run_summary("fake/path.csv")

        assert calls == ["fake/path.csv"]
        assert result_df.equals(df)
        assert "shape" in report
        assert report["shape"]["num_rows"] == 4
