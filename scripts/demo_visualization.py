from pathlib import Path

from unwrapped.clean import clean_data
from unwrapped.io import load_data
from unwrapped.visualization import (
    plot_audio_heatmap,
    plot_correlation_forest,
    plot_feature_correlations,
    plot_genre_popularity,
    plot_hit_vs_nonhit_profiles,
    plot_popularity_distribution,
    plot_top_genres,
    save_figure,
)


def main() -> None:
    data_path = "data/raw/spotify_data.csv"
    output_dir = Path("outputs")

    print("Loading and cleaning data...")
    raw_df = load_data(data_path)
    df, _ = clean_data(raw_df)
    print(f"  {len(df):,} tracks ready for visualization\n")

    fig1, _ = plot_top_genres(df, top_n=10)
    save_figure(fig1, output_dir / "top_genres.png")
    print("Saved outputs/top_genres.png")

    fig2, _ = plot_popularity_distribution(df, bins=20)
    save_figure(fig2, output_dir / "popularity_distribution.png")
    print("Saved outputs/popularity_distribution.png")

    fig3, _ = plot_feature_correlations(df)
    save_figure(fig3, output_dir / "feature_correlations.png")
    print("Saved outputs/feature_correlations.png")

    fig3b, _ = plot_correlation_forest(df)
    save_figure(fig3b, output_dir / "feature_correlations_forest.png")
    print("Saved outputs/feature_correlations_forest.png")

    fig4, _ = plot_genre_popularity(df, top_n=15)
    save_figure(fig4, output_dir / "genre_popularity.png")
    print("Saved outputs/genre_popularity.png")

    fig5, _ = plot_audio_heatmap(df)
    save_figure(fig5, output_dir / "audio_heatmap.png")
    print("Saved outputs/audio_heatmap.png")

    fig6, _ = plot_hit_vs_nonhit_profiles(df, threshold=70)
    save_figure(fig6, output_dir / "hit_vs_nonhit_profiles.png")
    print("Saved outputs/hit_vs_nonhit_profiles.png")


if __name__ == "__main__":
    main()
