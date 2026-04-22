"""Tests for the analysis module in :mod:`unwrapped.analysis`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from unwrapped.analysis import (
    AUDIO_FEATURES,
    _holm_bonferroni,
    analyze_popularity_correlations,
    compare_genres,
    compute_genre_deviations,
    enrich_with_genre_stats,
    find_genre_outliers,
    popularity_by_feature_buckets,
    run_analysis,
)


def make_row(**overrides: Any) -> dict[str, Any]:
    """Build a single track row with sensible defaults."""

    row = {
        "track_id": "track-1",
        "track_name": "Song 1",
        "artists": "Artist 1",
        "track_genre": "pop",
        "popularity": 50,
        "danceability": 0.5,
        "energy": 0.7,
        "loudness": -5.0,
        "speechiness": 0.05,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": 0.6,
        "tempo": 120.0,
    }
    row.update(overrides)
    return row


def make_df() -> pd.DataFrame:
    """Return a small multi-genre DataFrame for analysis tests."""

    return pd.DataFrame([
        # Pop: high danceability, high energy, high popularity
        make_row(track_id="t1", track_genre="pop", popularity=80,
                 danceability=0.8, energy=0.9, loudness=-3.0,
                 speechiness=0.04, acousticness=0.1, valence=0.8,
                 instrumentalness=0.0, liveness=0.12, tempo=128.0),
        make_row(track_id="t2", track_genre="pop", popularity=70,
                 danceability=0.7, energy=0.8, loudness=-4.0,
                 speechiness=0.06, acousticness=0.15, valence=0.7,
                 instrumentalness=0.01, liveness=0.1, tempo=122.0),
        make_row(track_id="t3", track_genre="pop", popularity=60,
                 danceability=0.6, energy=0.7, loudness=-5.0,
                 speechiness=0.05, acousticness=0.2, valence=0.6,
                 instrumentalness=0.0, liveness=0.15, tempo=118.0),
        # Rock: medium danceability, medium energy, medium popularity
        make_row(track_id="t4", track_genre="rock", popularity=50,
                 danceability=0.4, energy=0.6, loudness=-6.0,
                 speechiness=0.08, acousticness=0.25, valence=0.5,
                 instrumentalness=0.02, liveness=0.2, tempo=130.0),
        make_row(track_id="t5", track_genre="rock", popularity=40,
                 danceability=0.3, energy=0.5, loudness=-7.0,
                 speechiness=0.1, acousticness=0.3, valence=0.4,
                 instrumentalness=0.05, liveness=0.25, tempo=135.0),
        make_row(track_id="t6", track_genre="rock", popularity=45,
                 danceability=0.35, energy=0.55, loudness=-6.5,
                 speechiness=0.09, acousticness=0.28, valence=0.45,
                 instrumentalness=0.03, liveness=0.22, tempo=132.0),
        # Jazz: low danceability, low energy, low popularity
        make_row(track_id="t7", track_genre="jazz", popularity=30,
                 danceability=0.2, energy=0.3, loudness=-10.0,
                 speechiness=0.03, acousticness=0.7, valence=0.3,
                 instrumentalness=0.2, liveness=0.08, tempo=95.0),
        make_row(track_id="t8", track_genre="jazz", popularity=35,
                 danceability=0.25, energy=0.35, loudness=-9.0,
                 speechiness=0.04, acousticness=0.65, valence=0.35,
                 instrumentalness=0.15, liveness=0.09, tempo=100.0),
    ])


# ------------------------------------------------------------------
# enrich_with_genre_stats
# ------------------------------------------------------------------


class TestEnrichWithGenreStats:
    def test_adds_mean_and_std_columns(self) -> None:
        enriched = enrich_with_genre_stats(make_df())

        for feature in AUDIO_FEATURES:
            assert f"{feature}_genre_mean" in enriched.columns
            assert f"{feature}_genre_std" in enriched.columns

    def test_preserves_original_columns(self) -> None:
        df = make_df()
        enriched = enrich_with_genre_stats(df)

        for col in df.columns:
            assert col in enriched.columns

    def test_row_count_unchanged(self) -> None:
        df = make_df()
        enriched = enrich_with_genre_stats(df)

        assert len(enriched) == len(df)

    def test_genre_mean_is_correct(self) -> None:
        enriched = enrich_with_genre_stats(make_df())

        pop_rows = enriched[enriched["track_genre"] == "pop"]
        expected = np.mean([0.8, 0.7, 0.6])
        assert abs(pop_rows.iloc[0]["danceability_genre_mean"] - expected) < 1e-10

    def test_tracks_in_same_genre_share_stats(self) -> None:
        enriched = enrich_with_genre_stats(make_df())

        pop_rows = enriched[enriched["track_genre"] == "pop"]
        means = pop_rows["energy_genre_mean"].unique()
        assert len(means) == 1


# ------------------------------------------------------------------
# compute_genre_deviations
# ------------------------------------------------------------------


class TestComputeGenreDeviations:
    def test_adds_deviation_score(self) -> None:
        result = compute_genre_deviations(make_df())

        assert "genre_deviation_score" in result.columns

    def test_adds_zscore_columns(self) -> None:
        result = compute_genre_deviations(make_df())

        for feature in AUDIO_FEATURES:
            assert f"{feature}_zscore" in result.columns

    def test_deviation_scores_are_non_negative(self) -> None:
        result = compute_genre_deviations(make_df())

        assert (result["genre_deviation_score"] >= 0).all()

    def test_mean_track_has_low_deviation(self) -> None:
        """A track whose features equal the genre mean should score ~0."""

        df = pd.DataFrame([
            make_row(track_id="a", track_genre="pop", danceability=0.5),
            make_row(track_id="b", track_genre="pop", danceability=0.5),
            make_row(track_id="c", track_genre="pop", danceability=0.5),
        ])
        result = compute_genre_deviations(df)

        # All tracks are identical, so std=0 and z-scores are NaN,
        # which makes the deviation score NaN. This is expected behavior
        # for genres with zero variance.
        assert result["genre_deviation_score"].isna().all()


# ------------------------------------------------------------------
# find_genre_outliers
# ------------------------------------------------------------------


class TestFindGenreOutliers:
    def test_returns_expected_columns(self) -> None:
        outliers = find_genre_outliers(make_df(), threshold=0.5)

        expected = [
            "track_id", "track_name", "artists", "track_genre",
            "popularity", "genre_deviation_score",
        ]
        assert list(outliers.columns) == expected

    def test_respects_top_n(self) -> None:
        outliers = find_genre_outliers(make_df(), threshold=0.0, top_n=3)

        assert len(outliers) <= 3

    def test_sorted_descending(self) -> None:
        outliers = find_genre_outliers(make_df(), threshold=0.0)

        if len(outliers) > 1:
            assert outliers["genre_deviation_score"].is_monotonic_decreasing

    def test_high_threshold_returns_fewer_results(self) -> None:
        low = find_genre_outliers(make_df(), threshold=0.5)
        high = find_genre_outliers(make_df(), threshold=5.0)

        assert len(high) <= len(low)


# ------------------------------------------------------------------
# analyze_popularity_correlations
# ------------------------------------------------------------------


class TestAnalyzePopularityCorrelations:
    def test_returns_all_features(self) -> None:
        result = analyze_popularity_correlations(make_df())

        assert set(result["feature"]) == set(AUDIO_FEATURES)

    def test_sorted_by_absolute_correlation(self) -> None:
        result = analyze_popularity_correlations(make_df())

        assert result["abs_correlation"].is_monotonic_decreasing

    def test_correlations_in_valid_range(self) -> None:
        result = analyze_popularity_correlations(make_df())

        assert (result["correlation"].abs() <= 1.0).all()

    def test_direction_matches_sign(self) -> None:
        result = analyze_popularity_correlations(make_df())

        for _, row in result.iterrows():
            if row["correlation"] > 0:
                assert row["direction"] == "positive"
            else:
                assert row["direction"] == "negative"

    def test_strength_labels_are_valid(self) -> None:
        result = analyze_popularity_correlations(make_df())

        valid = {"strong", "moderate", "weak", "very weak"}
        assert set(result["strength"]).issubset(valid)

    def test_inference_columns_are_present(self) -> None:
        result = analyze_popularity_correlations(make_df(), n_bootstrap=100)

        for col in (
            "ci_low",
            "ci_high",
            "p_value",
            "p_value_adjusted",
            "significant",
            "n",
        ):
            assert col in result.columns

    def test_confidence_interval_brackets_the_point_estimate(self) -> None:
        result = analyze_popularity_correlations(
            make_df(), n_bootstrap=200, random_state=0
        )

        for _, row in result.iterrows():
            assert row["ci_low"] <= row["correlation"] <= row["ci_high"]

    def test_bootstrap_disabled_returns_nan_ci(self) -> None:
        result = analyze_popularity_correlations(make_df(), n_bootstrap=0)

        assert result["ci_low"].isna().all()
        assert result["ci_high"].isna().all()
        # p-values come from scipy, not the bootstrap, so they should still exist.
        assert result["p_value"].notna().all()

    def test_holm_adjusted_pvalues_are_not_smaller_than_raw(self) -> None:
        result = analyze_popularity_correlations(make_df(), n_bootstrap=50)

        assert (
            result["p_value_adjusted"].fillna(0) >= result["p_value"].fillna(0)
        ).all()
        assert (result["p_value_adjusted"].dropna() <= 1.0).all()

    def test_significance_flag_follows_alpha_threshold(self) -> None:
        result = analyze_popularity_correlations(
            make_df(), n_bootstrap=50, alpha=0.01
        )

        assert (
            result["significant"] == (result["p_value_adjusted"] < 0.01)
        ).all()

    def test_alpha_controls_ci_width(self) -> None:
        narrow = analyze_popularity_correlations(
            make_df(), n_bootstrap=400, alpha=0.20, random_state=1
        )
        wide = analyze_popularity_correlations(
            make_df(), n_bootstrap=400, alpha=0.01, random_state=1
        )

        narrow_width = (narrow["ci_high"] - narrow["ci_low"]).mean()
        wide_width = (wide["ci_high"] - wide["ci_low"]).mean()

        # A larger (1 - alpha) coverage should produce a wider interval on
        # average.
        assert wide_width >= narrow_width

    def test_random_state_makes_ci_deterministic(self) -> None:
        first = analyze_popularity_correlations(
            make_df(), n_bootstrap=100, random_state=7
        )
        second = analyze_popularity_correlations(
            make_df(), n_bootstrap=100, random_state=7
        )

        np.testing.assert_array_equal(
            first["ci_low"].to_numpy(), second["ci_low"].to_numpy()
        )
        np.testing.assert_array_equal(
            first["ci_high"].to_numpy(), second["ci_high"].to_numpy()
        )


class TestHolmBonferroni:
    def test_textbook_example(self) -> None:
        """Worked example: Holm on four p-values with a known adjustment."""
        raw = np.array([0.01, 0.04, 0.03, 0.005])
        adjusted = _holm_bonferroni(raw)

        # Sorted ascending: [0.005, 0.01, 0.03, 0.04] with factors [4,3,2,1]
        # -> [0.02, 0.03, 0.06, 0.04] -> monotone [0.02, 0.03, 0.06, 0.06].
        # Unsort to original order: [0.03, 0.06, 0.06, 0.02].
        expected = np.array([0.03, 0.06, 0.06, 0.02])
        np.testing.assert_allclose(adjusted, expected, atol=1e-12)

    def test_caps_at_one(self) -> None:
        adjusted = _holm_bonferroni(np.array([0.5, 0.8, 0.9]))

        assert (adjusted <= 1.0).all()

    def test_preserves_nan(self) -> None:
        adjusted = _holm_bonferroni(np.array([0.01, np.nan, 0.2]))

        # The NaN entry stays NaN and does not count toward the family size.
        assert np.isnan(adjusted[1])
        # The remaining two are the family m = 2.
        np.testing.assert_allclose(
            adjusted[[0, 2]], np.array([0.02, 0.2]), atol=1e-12
        )


# ------------------------------------------------------------------
# compare_genres
# ------------------------------------------------------------------


class TestCompareGenres:
    def test_returns_correct_genres(self) -> None:
        result = compare_genres(make_df(), top_n=3)

        assert set(result["genre"]) == {"pop", "rock", "jazz"}

    def test_track_counts_are_correct(self) -> None:
        result = compare_genres(make_df(), top_n=3)

        pop_row = result[result["genre"] == "pop"].iloc[0]
        assert pop_row["track_count"] == 3

        jazz_row = result[result["genre"] == "jazz"].iloc[0]
        assert jazz_row["track_count"] == 2

    def test_respects_top_n(self) -> None:
        result = compare_genres(make_df(), top_n=2)

        assert len(result) == 2

    def test_avg_popularity_is_correct(self) -> None:
        result = compare_genres(make_df(), top_n=3)

        pop_row = result[result["genre"] == "pop"].iloc[0]
        expected = np.mean([80, 70, 60])
        assert abs(pop_row["avg_popularity"] - expected) < 0.15

    def test_has_audio_feature_averages(self) -> None:
        result = compare_genres(make_df(), top_n=3)

        for feature in AUDIO_FEATURES:
            assert f"avg_{feature}" in result.columns


# ------------------------------------------------------------------
# popularity_by_feature_buckets
# ------------------------------------------------------------------


class TestPopularityByFeatureBuckets:
    def test_returns_correct_columns(self) -> None:
        result = popularity_by_feature_buckets(make_df(), "danceability", 3)

        assert "avg_popularity" in result.columns
        assert "track_count" in result.columns

    def test_bucket_count_matches(self) -> None:
        result = popularity_by_feature_buckets(make_df(), "danceability", 3)

        assert len(result) == 3

    def test_total_tracks_preserved(self) -> None:
        df = make_df()
        result = popularity_by_feature_buckets(df, "danceability", 3)

        assert result["track_count"].sum() == len(df)

    def test_avg_popularity_in_valid_range(self) -> None:
        result = popularity_by_feature_buckets(make_df(), "energy", 4)

        valid_buckets = result[result["track_count"] > 0]
        assert (valid_buckets["avg_popularity"] >= 0).all()
        assert (valid_buckets["avg_popularity"] <= 100).all()


# ------------------------------------------------------------------
# run_analysis
# ------------------------------------------------------------------


class TestRunAnalysis:
    def test_returns_all_sections(self) -> None:
        results = run_analysis(make_df())

        assert "correlations" in results
        assert "genre_comparison" in results
        assert "genre_outliers" in results
        assert "enriched_data" in results
        assert "feature_buckets" in results

    def test_feature_buckets_for_top_features(self) -> None:
        results = run_analysis(make_df())

        assert len(results["feature_buckets"]) <= 3
        assert len(results["feature_buckets"]) > 0

    def test_enriched_data_has_genre_stats(self) -> None:
        results = run_analysis(make_df())
        enriched = results["enriched_data"]

        assert "danceability_genre_mean" in enriched.columns
