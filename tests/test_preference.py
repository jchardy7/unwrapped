"""Tests for the preference scoring tool in :mod:`unwrapped.preference`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
    def test_roundtrip(self, tmp_path: Path) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")
        liked.add_by_id("t3")

        filepath = tmp_path / "likes.json"
        liked.save(str(filepath))

        liked2 = LikedSongs(make_df())
        liked2.load(str(filepath))

        assert liked2.liked_ids == {"t1", "t3"}

    def test_load_skips_unknown_ids(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        liked = LikedSongs(make_df())
        liked.add_by_id("t1")

        filepath = tmp_path / "likes.json"
        liked.save(str(filepath))

        # Load into a dataset that doesn't contain t1
        small_df = make_df().query("track_id != 't1'").reset_index(drop=True)
        liked2 = LikedSongs(small_df)
        liked2.load(str(filepath))

        assert "t1" not in liked2.liked_ids
        assert "Warning" in capsys.readouterr().out


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
