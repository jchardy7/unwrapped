from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_top_genres(df: pd.DataFrame, top_n: int = 10):
    """
    Plot the top N track genres by count.
    """
    if "track_genre" not in df.columns:
        raise ValueError("DataFrame must contain a 'track_genre' column.")

    genre_counts = (
        df["track_genre"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(top_n)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    genre_counts.plot(kind="barh", ax=ax)

    ax.set_title(f"Top {top_n} Track Genres")
    ax.set_xlabel("Number of Tracks")
    ax.set_ylabel("Genre")
    fig.tight_layout()

    return fig, ax


def plot_popularity_distribution(df: pd.DataFrame, bins: int = 20):
    """
    Plot a histogram of track popularity.
    """
    if "popularity" not in df.columns:
        raise ValueError("DataFrame must contain a 'popularity' column.")

    popularity = pd.to_numeric(df["popularity"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(popularity, bins=bins)

    ax.set_title("Distribution of Track Popularity")
    ax.set_xlabel("Popularity")
    ax.set_ylabel("Count")
    fig.tight_layout()

    return fig, ax


def save_figure(fig, output_path: str | Path) -> None:
    """
    Save a matplotlib figure to disk, creating parent directories if needed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")