import pandas as pd

from unwrapped.visualization import (
    plot_popularity_distribution,
    plot_top_genres,
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