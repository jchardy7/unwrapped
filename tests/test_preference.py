"""Tests for the preference scoring tool in :mod:`unwrapped.preference`."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import mock_open
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from unwrapped.preference import AUDIO_FEATURES, LikedSongs


def make_row(**overrides: Any) -> dict[str, Any]:
    """Build a single row with all columns required by LikedSongs."""
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
    return pd.DataFrame([
        make_row(track_id="t1", track_name="Song A", artists="Artist A",
                 danceability=0.3, energy=0.4, tempo=100.0),
        make_row(track_id="t2", track_name="Song B", artists="Artist B",
                 danceability=0.7, energy=0.9, tempo=140.0),
        make_row(track_id="t3", track_name="Song C", artists="Artist A",
                 danceability=0.5, energy=0.6, tempo=120.0),
        make_row(track_id="t4", track_name="Song D", artists="Artist C",
                 danceability=0.9, energy=0.3, tempo=90.0),
    ])


class TestAddSongs:
    def test_add_by_id(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        assert "t1" in liked.liked_ids
        assert len(liked) == 1

    def test_add_by_id_raises_for_unknown_id(self) -> None:
        liked = LikedSongs(make_df())

        with pytest.raises(ValueError, match="not found"):
            liked.add_by_id("nonexistent")

    def test_add_by_name(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_name("Song B")

        assert "t2" in liked.liked_ids

    def test_add_by_name_case_insensitive(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_name("song b")

        assert "t2" in liked.liked_ids

    def test_add_by_name_with_artist(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_name("Song A", artist="Artist A")

        assert "t1" in liked.liked_ids

    def test_add_by_name_raises_for_unknown_name(self) -> None:
        liked = LikedSongs(make_df())

        with pytest.raises(ValueError, match="No track named"):
            liked.add_by_name("Nonexistent Song")

    def test_remove(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")
        liked.remove("t1")

        assert len(liked) == 0

    def test_clear(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")
        liked.add_by_id("t2")
        liked.clear()

        assert len(liked) == 0


class TestShow:
    def test_show_returns_liked_tracks(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")
        liked.add_by_id("t3")

        result = liked.show()

        assert len(result) == 2
        assert set(result["track_id"]) == {"t1", "t3"}
        assert list(result.columns) == ["track_id", "track_name", "artists", "popularity"]

    def test_show_empty_returns_empty_df(self) -> None:
        liked = LikedSongs(make_df())
        result = liked.show()

        assert result.empty


class TestPredict:
    def test_predict_returns_zero_scores_when_all_tracks_match_profile(self) -> None:
        df = pd.DataFrame([
            make_row(track_id="t1", track_name="Song A", artists="Artist A"),
            make_row(track_id="t2", track_name="Song B", artists="Artist B"),
            make_row(track_id="t3", track_name="Song C", artists="Artist C"),
        ])
        liked = LikedSongs(df)
        liked.add_by_id("t1")

        scores = liked.predict()

        assert set(scores["track_id"]) == {"t2", "t3"}
        assert (scores["preference_score"] == 0.0).all()

    def test_predict_ranks_single_liked_track_first_when_included(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        scores = liked.predict(top_n=1, exclude_liked=False)

        assert scores.iloc[0]["track_id"] == "t1"
        assert scores.iloc[0]["preference_score"] == pytest.approx(1.0)

    def test_predict_returns_scores_in_range(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        scores = liked.predict()

        assert "preference_score" in scores.columns
        assert scores["preference_score"].min() >= 0.0
        assert scores["preference_score"].max() <= 1.0

    def test_predict_sorted_descending(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        scores = liked.predict()

        assert scores["preference_score"].is_monotonic_decreasing

    def test_predict_excludes_liked_by_default(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        scores = liked.predict()

        assert "t1" not in scores["track_id"].values

    def test_predict_includes_liked_when_requested(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        scores = liked.predict(exclude_liked=False)

        assert "t1" in scores["track_id"].values

    def test_predict_top_n(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        scores = liked.predict(top_n=2)

        assert len(scores) == 2

    def test_predict_raises_when_empty(self) -> None:
        liked = LikedSongs(make_df())

        with pytest.raises(ValueError, match="empty"):
            liked.predict()

    def test_predict_output_columns(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        scores = liked.predict()

        expected_cols = ["track_id", "track_name", "artists", "track_genre",
                         "popularity", "preference_score"]
        assert list(scores.columns) == expected_cols


class TestBuildProfile:
    def test_profile_raises_when_empty(self) -> None:
        liked = LikedSongs(make_df())

        with pytest.raises(ValueError, match="empty"):
            liked.build_profile()

    def test_profile_returns_mean_of_liked(self) -> None:
        df = make_df()
        liked = LikedSongs(df)
        liked.add_by_id("t1")
        liked.add_by_id("t2")

        profile = liked.build_profile()

        t1 = df[df["track_id"] == "t1"].iloc[0]
        t2 = df[df["track_id"] == "t2"].iloc[0]
        for feat in AUDIO_FEATURES:
            expected = (t1[feat] + t2[feat]) / 2
            assert abs(profile[feat] - expected) < 1e-10


class TestSaveLoad:
    def test_save_creates_parent_dirs_and_writes_sorted_ids(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t3")
        liked.add_by_id("t1")

        with patch("pathlib.Path.mkdir") as mkdir_mock, patch(
            "pathlib.Path.open", mock_open()
        ), patch("unwrapped.preference.json.dump") as dump_mock:
            liked.save("nested/likes.json")

        mkdir_mock.assert_called_once_with(parents=True, exist_ok=True)
        assert dump_mock.call_args.args[0] == ["t1", "t3"]
        assert dump_mock.call_args.kwargs["indent"] == 2

    def test_load_adds_known_ids_from_json_list(self) -> None:
        liked = LikedSongs(make_df())

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.open", mock_open(read_data='["t1", "t3"]')
        ):
            liked.load("likes.json")

        assert liked.liked_ids == {"t1", "t3"}

    def test_load_skips_unknown_ids(self, capsys: pytest.CaptureFixture[str]) -> None:
        small_df = make_df().query("track_id != 't1'").reset_index(drop=True)
        liked = LikedSongs(small_df)

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.open", mock_open(read_data='["t1"]')
        ):
            liked.load("likes.json")

        assert "t1" not in liked.liked_ids
        assert "Warning" in capsys.readouterr().out

    def test_load_raises_when_json_payload_is_not_a_list(self) -> None:
        liked = LikedSongs(make_df())

        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.open", mock_open(read_data=json.dumps({"track_id": "t1"}))
        ):
            with pytest.raises(ValueError, match="must contain a JSON list"):
                liked.load("likes.json")


class TestValidation:
    def test_raises_for_missing_columns(self) -> None:
        df = pd.DataFrame({"track_id": ["t1"], "track_name": ["Song"]})

        with pytest.raises(ValueError, match="missing required columns"):
            LikedSongs(df)

    def test_deduplicates_track_ids(self) -> None:
        df = make_df()
        dup = pd.concat([df, df.head(1)], ignore_index=True)
        liked = LikedSongs(dup)

        assert liked.df["track_id"].is_unique


class TestPredictEuclidean:
    def test_scores_in_range(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        scores = liked.predict(method="euclidean")

        assert scores["preference_score"].min() >= 0.0
        assert scores["preference_score"].max() <= 1.0

    def test_sorted_descending(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        scores = liked.predict(method="euclidean")

        assert scores["preference_score"].is_monotonic_decreasing

    def test_inverse_variance_needs_two_liked_songs(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        with pytest.warns(UserWarning, match="at least 2 liked songs"):
            liked.predict(method="euclidean", weights="inverse_variance")

    def test_inverse_variance_differs_from_uniform(self) -> None:
        # Construct liked songs that are consistent on danceability but
        # inconsistent on energy, so inverse-variance weighting should
        # reweight features asymmetrically relative to uniform.
        rng = np.random.default_rng(0)
        rows = []
        for i in range(15):
            rows.append(
                make_row(
                    track_id=f"tv{i}",
                    track_name=f"Song {i}",
                    danceability=float(rng.uniform(0.1, 0.9)),
                    energy=float(rng.uniform(0.1, 0.9)),
                    tempo=float(rng.uniform(80, 160)),
                    loudness=float(rng.uniform(-12, -2)),
                    valence=float(rng.uniform(0.1, 0.9)),
                )
            )
        df = pd.DataFrame(rows)
        # Liked songs agree on danceability ≈ 0.5 but disagree on energy.
        df.loc[0, "danceability"] = 0.50
        df.loc[1, "danceability"] = 0.51
        df.loc[2, "danceability"] = 0.49
        df.loc[0, "energy"] = 0.10
        df.loc[1, "energy"] = 0.80
        df.loc[2, "energy"] = 0.30

        liked = LikedSongs(df)
        liked.add_by_id("tv0")
        liked.add_by_id("tv1")
        liked.add_by_id("tv2")

        uniform = liked.predict(method="euclidean", weights="uniform")
        weighted = liked.predict(method="euclidean", weights="inverse_variance")

        merged = uniform.merge(
            weighted, on="track_id", suffixes=("_u", "_w")
        )
        assert not np.allclose(
            merged["preference_score_u"], merged["preference_score_w"]
        )

    def test_unknown_method_raises(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        with pytest.raises(ValueError, match="Unknown method"):
            liked.predict(method="manhattan")  # type: ignore[arg-type]

    def test_unknown_weights_raises(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")
        liked.add_by_id("t2")

        with pytest.raises(ValueError, match="Unknown weights"):
            liked.predict(
                method="euclidean",
                weights="tf-idf",  # type: ignore[arg-type]
            )


class TestPredictExplanations:
    def test_top_matches_column_added(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        scores = liked.predict(return_explanations=True)

        assert "top_matches" in scores.columns
        # Each row should list 3 feature names by default.
        first = scores.iloc[0]["top_matches"]
        assert len(first.split(", ")) == 3

    def test_explanations_dont_change_base_scores(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        baseline = liked.predict()
        with_explanations = liked.predict(return_explanations=True)

        pd.testing.assert_series_equal(
            baseline["preference_score"],
            with_explanations["preference_score"],
        )


class TestExplain:
    def test_returns_row_per_feature(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        result = liked.explain("t2")

        assert len(result) == len(AUDIO_FEATURES)
        assert set(result["feature"]) == set(AUDIO_FEATURES)

    def test_expected_columns_for_cosine(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        result = liked.explain("t2", method="cosine")

        expected = {
            "feature", "track_raw", "profile_raw",
            "track_scaled", "profile_scaled",
            "delta", "abs_delta", "weight",
        }
        assert set(result.columns) == expected

    def test_euclidean_includes_attribution(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        result = liked.explain("t2", method="euclidean")

        assert "attribution" in result.columns
        # All euclidean attributions are non-positive (squared deltas).
        assert (result["attribution"] <= 0).all()

    def test_sorted_by_abs_delta_ascending(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")
        liked.add_by_id("t2")

        result = liked.explain("t3", method="euclidean")

        assert result["abs_delta"].is_monotonic_increasing

    def test_euclidean_attributions_sum_to_raw_score(self) -> None:
        """sum(attribution) must equal the raw euclidean score exactly."""
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")
        liked.add_by_id("t2")

        # Recompute the raw score directly for t3 and compare.
        explanation = liked.explain("t3", method="euclidean")
        attribution_sum = explanation["attribution"].sum()

        # The raw score from _raw_scores uses the same formula; compute it
        # inline so the test doesn't depend on internal caching.
        scaled_matrix, profile_scaled = liked._scaled_feature_space()
        weights = liked._compute_weights("uniform")
        deltas = scaled_matrix - profile_scaled
        all_raw = -np.sum(weights * deltas ** 2, axis=1)
        t3_idx = liked.df.index[liked.df["track_id"] == "t3"][0]
        expected_raw = all_raw[t3_idx]

        assert attribution_sum == pytest.approx(expected_raw, abs=1e-12)

    def test_raises_for_unknown_track(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        with pytest.raises(ValueError, match="not found"):
            liked.explain("nonexistent")

    def test_raises_when_liked_empty(self) -> None:
        liked = LikedSongs(make_df())

        with pytest.raises(ValueError, match="empty"):
            liked.explain("t1")


class TestExplainTop:
    def test_returns_matches_and_mismatches(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        top = liked.explain_top("t2", n=3)

        assert set(top.keys()) == {"matches", "mismatches"}
        assert len(top["matches"]) == 3
        assert len(top["mismatches"]) == 3

    def test_matches_have_smaller_abs_delta_than_mismatches(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        top = liked.explain_top("t2", n=2)

        assert top["matches"]["abs_delta"].max() <= top["mismatches"]["abs_delta"].min()

    def test_mismatches_sorted_worst_first(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        top = liked.explain_top("t2", n=3)

        assert top["mismatches"]["abs_delta"].is_monotonic_decreasing

    def test_rejects_non_positive_n(self) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        with pytest.raises(ValueError, match="n must be"):
            liked.explain_top("t2", n=0)
