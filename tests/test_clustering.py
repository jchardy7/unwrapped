from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from unwrapped.clustering import (
    cluster_centroids,
    cluster_songs,
    cluster_summary,
    find_optimal_k,
    prepare_clustering_data,
    run_clustering_pipeline,
    save_outputs,
)


def make_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "track_id": "t",
        "artists": "Artist",
        "album_name": "Album",
        "track_name": "Song",
        "track_genre": "pop",
        "popularity": 50,
        "duration_ms": 200000,
        "explicit": 0,
        "danceability": 0.6,
        "energy": 0.7,
        "key": 5,
        "loudness": -5.0,
        "mode": 1,
        "speechiness": 0.05,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": 0.5,
        "tempo": 120.0,
        "time_signature": 4,
    }
    row.update(overrides)
    return row


def make_df() -> pd.DataFrame:
    """Three well-separated clusters of 8 rows each."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(42)

    centers = [
        {"danceability": 0.85, "energy": 0.85, "valence": 0.8, "acousticness": 0.05,
         "tempo": 130, "popularity": 80, "track_genre": "pop"},
        {"danceability": 0.4, "energy": 0.85, "valence": 0.4, "acousticness": 0.05,
         "tempo": 145, "popularity": 50, "track_genre": "rock"},
        {"danceability": 0.3, "energy": 0.2, "valence": 0.5, "acousticness": 0.85,
         "tempo": 90, "popularity": 35, "track_genre": "acoustic"},
    ]
    for cluster_idx, c in enumerate(centers):
        for i in range(8):
            jitter = rng.normal(0, 0.01)
            rows.append(
                make_row(
                    track_id=f"c{cluster_idx}-{i}",
                    track_genre=c["track_genre"],
                    popularity=c["popularity"],
                    danceability=c["danceability"] + jitter,
                    energy=c["energy"] + jitter,
                    valence=c["valence"] + jitter,
                    acousticness=c["acousticness"] + jitter,
                    tempo=c["tempo"] + rng.normal(0, 0.5),
                )
            )
    return pd.DataFrame(rows)


def test_prepare_clustering_data_returns_zero_mean_unit_variance():
    df = make_df()
    X_scaled, scaler, feature_names, kept_index = prepare_clustering_data(df)

    assert X_scaled.shape == (len(df), len(feature_names))
    np.testing.assert_allclose(X_scaled.mean(axis=0), 0.0, atol=1e-9)
    # Each scaled column should be either ~0 (constant input) or ~1 (varying);
    # sklearn's StandardScaler leaves zero-variance columns at 0.
    scaled_std = X_scaled.std(axis=0)
    near_zero_or_one = np.isclose(scaled_std, 0.0, atol=1e-9) | np.isclose(
        scaled_std, 1.0, atol=1e-9
    )
    assert near_zero_or_one.all()
    # At least some columns must be nontrivially scaled.
    assert np.isclose(scaled_std, 1.0, atol=1e-9).any()
    assert list(kept_index) == list(df.index)


def test_prepare_clustering_data_drops_rows_missing_features():
    df = make_df()
    df.loc[0, "danceability"] = np.nan

    X_scaled, _, _, kept_index = prepare_clustering_data(df)

    assert len(X_scaled) == len(df) - 1
    assert 0 not in kept_index


def test_prepare_clustering_data_raises_on_missing_columns():
    df = make_df().drop(columns=["energy"])
    with pytest.raises(ValueError, match="Missing required columns"):
        prepare_clustering_data(df)


def test_prepare_clustering_data_accepts_custom_feature_list():
    df = make_df()
    X_scaled, _, feature_names, _ = prepare_clustering_data(
        df, features=["danceability", "energy"]
    )

    assert feature_names == ["danceability", "energy"]
    assert X_scaled.shape[1] == 2


def test_find_optimal_k_returns_required_columns():
    df = make_df()
    X_scaled, _, _, _ = prepare_clustering_data(df)

    result = find_optimal_k(X_scaled, k_range=range(2, 5))

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["k", "inertia", "silhouette"]
    assert list(result["k"]) == [2, 3, 4]


def test_find_optimal_k_inertia_is_non_increasing():
    df = make_df()
    X_scaled, _, _, _ = prepare_clustering_data(df)

    result = find_optimal_k(X_scaled, k_range=range(2, 6))

    inertias = result["inertia"].to_list()
    assert all(a >= b - 1e-9 for a, b in zip(inertias, inertias[1:]))


def test_cluster_songs_adds_cluster_column():
    df = make_df()
    df_clustered, model, scaler = cluster_songs(df, n_clusters=3)

    assert "cluster" in df_clustered.columns
    assert len(df_clustered) == len(df)
    assigned = df_clustered.loc[df_clustered["cluster"] >= 0, "cluster"].unique()
    assert set(assigned) == {0, 1, 2}
    assert hasattr(model, "cluster_centers_")


def test_cluster_songs_marks_rows_with_missing_features_as_minus_one():
    df = make_df()
    df.loc[0, "danceability"] = np.nan

    df_clustered, _, _ = cluster_songs(df, n_clusters=3)

    assert df_clustered.loc[0, "cluster"] == -1


def test_cluster_centroids_shape_and_range():
    df = make_df()
    df_clustered, model, scaler = cluster_songs(df, n_clusters=3)
    _, _, feature_names, _ = prepare_clustering_data(df)

    centroids = cluster_centroids(model, scaler, feature_names)

    assert centroids.shape == (3, len(feature_names))
    assert list(centroids.columns) == feature_names
    for feat in feature_names:
        feat_min, feat_max = df[feat].min(), df[feat].max()
        spread = feat_max - feat_min
        # Centroids should sit within the data range (allow a tiny float margin).
        assert centroids[feat].min() >= feat_min - 1e-6 * max(1.0, spread)
        assert centroids[feat].max() <= feat_max + 1e-6 * max(1.0, spread)


def test_cluster_summary_one_row_per_cluster():
    df = make_df()
    df_clustered, _, _ = cluster_songs(df, n_clusters=3)

    summary = cluster_summary(df_clustered)

    assert len(summary) == 3
    assert {"cluster", "size", "mean_popularity", "top_genre"} <= set(summary.columns)
    assert summary["size"].sum() == len(df)
    assert summary["cluster"].is_monotonic_increasing


def test_cluster_summary_has_one_mean_column_per_audio_feature():
    df = make_df()
    df_clustered, _, _ = cluster_songs(df, n_clusters=2)

    summary = cluster_summary(df_clustered)

    for feat in ["danceability", "energy", "valence", "acousticness", "tempo"]:
        assert f"mean_{feat}" in summary.columns


def test_save_outputs_writes_expected_files(tmp_path: Path):
    assignments = pd.DataFrame({"track_id": ["a"], "cluster": [0]})
    centroids = pd.DataFrame([[0.5, 0.7]], columns=["danceability", "energy"])
    summary = pd.DataFrame({"cluster": [0], "size": [1]})
    k_search = pd.DataFrame({"k": [2, 3], "inertia": [10.0, 8.0], "silhouette": [0.5, 0.6]})

    paths = save_outputs(
        assignments, centroids, summary, k_search, output_dir=tmp_path / "out"
    )

    assert {"assignments", "centroids", "summary", "k_search"} <= paths.keys()
    assert (tmp_path / "out" / "cluster_assignments.csv").exists()
    assert (tmp_path / "out" / "cluster_centroids.csv").exists()
    assert (tmp_path / "out" / "cluster_summary.csv").exists()
    assert (tmp_path / "out" / "cluster_k_search.csv").exists()


def test_save_outputs_omits_k_search_when_none(tmp_path: Path):
    assignments = pd.DataFrame({"track_id": ["a"], "cluster": [0]})
    centroids = pd.DataFrame([[0.5]], columns=["danceability"])
    summary = pd.DataFrame({"cluster": [0], "size": [1]})

    paths = save_outputs(assignments, centroids, summary, None, output_dir=tmp_path / "out")

    assert "k_search" not in paths
    assert not (tmp_path / "out" / "cluster_k_search.csv").exists()


def test_run_clustering_pipeline_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    df = make_df()
    monkeypatch.setattr("unwrapped.clustering.load_data", lambda _: df)

    output_dir = tmp_path / "pipeline"
    result = run_clustering_pipeline(
        data_path="fake.csv",
        n_clusters=3,
        k_search_range=range(2, 5),
        save_results=True,
        output_dir=output_dir,
    )

    assert {"model", "scaler", "assignments", "centroids", "summary", "k_search"} <= result.keys()
    assert len(result["centroids"]) == 3
    assert len(result["summary"]) == 3
    assert (output_dir / "cluster_assignments.csv").exists()
    assert (output_dir / "cluster_summary.csv").exists()
    assert (output_dir / "cluster_k_search.csv").exists()
