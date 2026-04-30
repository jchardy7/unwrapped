"""Run the full analysis pipeline and present findings.

This script cleans the raw Spotify dataset, then answers three research
questions using the analysis module.  Results are printed to the terminal
in a readable format with key takeaways highlighted.
"""

import sys

from unwrapped.io import DEFAULT_DATA_PATH, load_data
from unwrapped.clean import clean_data
from unwrapped.analysis import run_analysis

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def print_header(title):
    """Print a prominent section header."""
    print()
    print("=" * 64)
    print(f"  {title}")
    print("=" * 64)


def print_section(title):
    """Print a subsection header."""
    print()
    print("-" * 64)
    print(f"  {title}")
    print("-" * 64)
    print()


def main():
    # Load and clean the data before analysis
    raw_df = load_data(DEFAULT_DATA_PATH)
    df, cleaning_report = clean_data(raw_df)

    print_header("Unwrapped: Spotify Dataset Analysis")

    print(f"\nDataset size after cleaning: {len(df):,} tracks")
    print(f"Rows removed during cleaning: {cleaning_report['rows_removed_total']:,}")
    print(f"Unique genres: {df['track_genre'].nunique()}")
    print(f"Unique artists: {df['artists'].nunique()}")

    # Run the full analysis
    results = run_analysis(df)

    # ==================================================================
    # Research Question 1
    # ==================================================================

    print_section(
        "Q1: Which audio features are most associated with popularity?"
    )

    corr = results["correlations"]

    display = corr.copy()
    display["95% CI"] = display.apply(
        lambda row: f"[{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]", axis=1
    )
    display["p (Holm)"] = display["p_value_adjusted"].map(
        lambda p: "<1e-4" if p < 1e-4 else f"{p:.4f}"
    )
    display["sig"] = display["significant"].map(lambda s: "yes" if s else "no")
    print(
        display[
            ["feature", "correlation", "95% CI", "p (Holm)", "sig", "strength"]
        ].to_string(index=False)
    )

    top = corr.iloc[0]
    second = corr.iloc[1]
    third = corr.iloc[2]

    print(f"\nFindings:")
    print(
        f"  The strongest predictor of popularity is {top['feature']} "
        f"(r = {top['correlation']:+.4f}, 95% CI "
        f"[{top['ci_low']:+.3f}, {top['ci_high']:+.3f}], "
        f"Holm-adj. p = "
        f"{'<1e-4' if top['p_value_adjusted'] < 1e-4 else f'{top['p_value_adjusted']:.4f}'}"
        f"), followed by "
        f"{second['feature']} (r = {second['correlation']:+.4f}) and "
        f"{third['feature']} (r = {third['correlation']:+.4f})."
    )

    n_significant = int(corr["significant"].sum())
    print(
        f"  {n_significant} of {len(corr)} features stay significant "
        f"after Holm–Bonferroni correction."
    )

    # Check direction patterns
    positive = corr[corr["direction"] == "positive"]["feature"].tolist()
    negative = corr[corr["direction"] == "negative"]["feature"].tolist()

    if positive:
        print(
            f"  Features with positive correlation (higher = more popular): "
            f"{', '.join(positive)}"
        )
    if negative:
        print(
            f"  Features with negative correlation (higher = less popular): "
            f"{', '.join(negative)}"
        )

    # Show bucket breakdown for the strongest feature
    print(
        f"\n  Breaking {top['feature']} into buckets to look for "
        f"non-linear patterns:"
    )
    top_buckets = results["feature_buckets"][top["feature"]]
    print()
    print(top_buckets.to_string())

    # ==================================================================
    # Research Question 2
    # ==================================================================

    print_section("Q2: How do genres differ in their audio profiles?")

    genre_comp = results["genre_comparison"]

    display_cols = [
        "genre",
        "track_count",
        "avg_popularity",
        "avg_danceability",
        "avg_energy",
        "avg_acousticness",
        "avg_valence",
        "avg_tempo",
    ]
    available = [c for c in display_cols if c in genre_comp.columns]
    print(genre_comp[available].to_string(index=False))

    # Pull out some interesting contrasts
    most_dance = genre_comp.loc[genre_comp["avg_danceability"].idxmax()]
    least_dance = genre_comp.loc[genre_comp["avg_danceability"].idxmin()]
    most_energy = genre_comp.loc[genre_comp["avg_energy"].idxmax()]
    least_energy = genre_comp.loc[genre_comp["avg_energy"].idxmin()]
    most_popular = genre_comp.loc[genre_comp["avg_popularity"].idxmax()]
    least_popular = genre_comp.loc[genre_comp["avg_popularity"].idxmin()]
    most_acoustic = genre_comp.loc[genre_comp["avg_acousticness"].idxmax()]

    print(f"\nFindings:")
    print(
        f"  Most danceable genre: {most_dance['genre']} "
        f"(avg = {most_dance['avg_danceability']:.2f})"
    )
    print(
        f"  Least danceable genre: {least_dance['genre']} "
        f"(avg = {least_dance['avg_danceability']:.2f})"
    )
    print(
        f"  Most energetic genre: {most_energy['genre']} "
        f"(avg = {most_energy['avg_energy']:.2f})"
    )
    print(
        f"  Least energetic genre: {least_energy['genre']} "
        f"(avg = {least_energy['avg_energy']:.2f})"
    )
    print(
        f"  Most acoustic genre: {most_acoustic['genre']} "
        f"(avg = {most_acoustic['avg_acousticness']:.2f})"
    )
    print(
        f"  Highest avg popularity: {most_popular['genre']} "
        f"({most_popular['avg_popularity']:.1f})"
    )
    print(
        f"  Lowest avg popularity: {least_popular['genre']} "
        f"({least_popular['avg_popularity']:.1f})"
    )

    # ==================================================================
    # Research Question 3
    # ==================================================================

    print_section("Q3: Which tracks don't fit their genre?")

    outliers = results["genre_outliers"]

    if outliers.empty:
        print("No strong genre outliers found with the default threshold.")
    else:
        print(f"Found {len(outliers)} tracks with deviation score > 2.0:\n")
        print(outliers.to_string(index=False))

        top_outlier = outliers.iloc[0]
        print(
            f"\nFindings:"
            f"\n  The biggest outlier is \"{top_outlier['track_name']}\" by "
            f"{top_outlier['artists']} in the {top_outlier['track_genre']} "
            f"genre (deviation score = "
            f"{top_outlier['genre_deviation_score']:.2f})."
            f"\n  This means its audio features are very different from the "
            f"typical {top_outlier['track_genre']} track."
        )

    # ==================================================================
    # Summary
    # ==================================================================

    print()
    print("=" * 64)
    print("  Analysis complete.")
    print("=" * 64)
    print()


if __name__ == "__main__":
    main()
