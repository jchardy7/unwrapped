from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from unwrapped.visualization import (
    plot_actual_vs_predicted,
    plot_audio_heatmap,
    plot_correlation_forest,
    plot_feature_correlations,
    plot_genre_popularity,
    plot_hit_vs_nonhit_profiles,
    plot_popularity_distribution,
    plot_top_genres,
    save_figure,
)


def test_plot_top_genres_runs():
    df = pd.DataFrame(
        {
            "track_genre": ["pop", "pop", "rock", "jazz", "rock", "pop"],
            "popularity": [90, 80, 75, 60, 70, 95],
        }
    )

    fig, ax = plot_top_genres(df, top_n=2)

    assert fig is not None
    assert ax is not None


def test_plot_popularity_distribution_runs():
    df = pd.DataFrame(
        {
            "track_genre": ["pop", "rock", "jazz"],
            "popularity": [90, 70, 50],
        }
    )

    fig, ax = plot_popularity_distribution(df, bins=5)

    assert fig is not None
    assert ax is not None


def test_plot_top_genres_raises_when_genre_column_missing():
    df = pd.DataFrame({"popularity": [90, 80, 75]})

    with pytest.raises(ValueError, match="track_genre"):
        plot_top_genres(df)


def test_plot_popularity_distribution_raises_when_popularity_missing():
    df = pd.DataFrame({"track_genre": ["pop", "rock"]})

    with pytest.raises(ValueError, match="popularity"):
        plot_popularity_distribution(df)


def _sample_df():
    return pd.DataFrame(
        {
            "track_genre": ["pop", "pop", "rock", "jazz", "rock", "pop", "jazz", "rock"],
            "popularity": [90, 80, 75, 60, 70, 95, 40, 55],
            "danceability": [0.8, 0.7, 0.5, 0.4, 0.6, 0.9, 0.3, 0.5],
            "energy": [0.9, 0.8, 0.7, 0.3, 0.8, 0.85, 0.2, 0.75],
            "loudness": [-4.0, -5.0, -6.0, -12.0, -5.5, -3.0, -15.0, -7.0],
            "speechiness": [0.05, 0.04, 0.06, 0.03, 0.07, 0.05, 0.02, 0.06],
            "acousticness": [0.1, 0.2, 0.3, 0.8, 0.2, 0.05, 0.9, 0.25],
            "instrumentalness": [0.0, 0.0, 0.1, 0.5, 0.0, 0.0, 0.7, 0.05],
            "liveness": [0.1, 0.2, 0.15, 0.12, 0.18, 0.1, 0.11, 0.16],
            "valence": [0.7, 0.6, 0.5, 0.4, 0.65, 0.8, 0.3, 0.55],
            "tempo": [120.0, 110.0, 130.0, 90.0, 125.0, 118.0, 85.0, 132.0],
        }
    )


def test_plot_feature_correlations_runs():
    fig, ax = plot_feature_correlations(_sample_df())
    assert fig is not None
    assert ax is not None


def test_plot_feature_correlations_raises_on_missing_column():
    df = pd.DataFrame({"popularity": [90, 80]})
    with pytest.raises(ValueError, match="missing columns"):
        plot_feature_correlations(df)


def test_plot_correlation_forest_runs():
    fig, ax = plot_correlation_forest(
        _sample_df(), n_bootstrap=50, random_state=0
    )
    assert fig is not None
    assert ax is not None


def test_plot_genre_popularity_runs():
    fig, ax = plot_genre_popularity(_sample_df(), top_n=3)
    assert fig is not None
    assert ax is not None


def test_plot_genre_popularity_raises_on_missing_column():
    df = pd.DataFrame({"track_genre": ["pop", "rock"]})
    with pytest.raises(ValueError, match="popularity"):
        plot_genre_popularity(df)


def test_plot_audio_heatmap_runs():
    fig, ax = plot_audio_heatmap(_sample_df())
    assert fig is not None
    assert ax is not None


def test_plot_audio_heatmap_raises_on_missing_column():
    df = pd.DataFrame({"danceability": [0.5, 0.6]})
    with pytest.raises(ValueError, match="missing columns"):
        plot_audio_heatmap(df)


def test_plot_hit_vs_nonhit_profiles_runs():
    fig, ax = plot_hit_vs_nonhit_profiles(_sample_df(), threshold=70)
    assert fig is not None
    assert ax is not None


def test_plot_hit_vs_nonhit_profiles_raises_on_missing_column():
    df = pd.DataFrame({"popularity": [90, 80]})
    with pytest.raises(ValueError, match="missing columns"):
        plot_hit_vs_nonhit_profiles(df)


def test_plot_actual_vs_predicted_runs():
    predictions_df = pd.DataFrame(
        {
            "actual_popularity": [80, 60, 70, 50, 90, 40, 75, 55],
            "linear_prediction": [75.0, 65.0, 72.0, 48.0, 88.0, 42.0, 74.0, 58.0],
            "random_forest_prediction": [78.0, 62.0, 71.0, 52.0, 87.0, 38.0, 76.0, 54.0],
        }
    )

    fig, axes = plot_actual_vs_predicted(predictions_df)

    assert fig is not None
    assert len(axes) == 2


def test_plot_actual_vs_predicted_raises_on_missing_column():
    df = pd.DataFrame(
        {
            "actual_popularity": [80, 60],
            "linear_prediction": [75.0, 65.0],
        }
    )

    with pytest.raises(ValueError, match="missing columns"):
        plot_actual_vs_predicted(df)


def test_plot_actual_vs_predicted_titles_contain_r2():
    predictions_df = pd.DataFrame(
        {
            "actual_popularity": [80, 60, 70, 50, 90],
            "linear_prediction": [75.0, 65.0, 72.0, 48.0, 88.0],
            "random_forest_prediction": [78.0, 62.0, 71.0, 52.0, 87.0],
        }
    )

    _, axes = plot_actual_vs_predicted(predictions_df)

    for ax in axes:
        assert "R²" in ax.get_title()


def test_save_figure_writes_file_and_creates_parent_dirs(tmp_path: Path):
    df = pd.DataFrame(
        {
            "track_genre": ["pop", "rock", "jazz"],
            "popularity": [90, 70, 50],
        }
    )
    fig, _ = plot_popularity_distribution(df, bins=3)

    output_path = tmp_path / "nested" / "subdir" / "figure.png"
    save_figure(fig, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
