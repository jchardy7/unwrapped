from pathlib import Path

from unwrapped.io import load_tracks
from unwrapped.visualization import (
    plot_popularity_distribution,
    plot_top_genres,
    save_figure,
)


def main() -> None:
    data_path = "data/raw/spotify_data.csv"
    output_dir = Path("outputs")

    df = load_tracks(data_path)

    fig1, _ = plot_top_genres(df, top_n=10)
    save_figure(fig1, output_dir / "top_genres.png")
    print("Saved outputs/top_genres.png")

    fig2, _ = plot_popularity_distribution(df, bins=20)
    save_figure(fig2, output_dir / "popularity_distribution.png")
    print("Saved outputs/popularity_distribution.png")


if __name__ == "__main__":
    main()