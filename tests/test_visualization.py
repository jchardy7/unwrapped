from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from unwrapped.visualization import (
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
