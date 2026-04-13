"""Analysis module for exploring the Spotify dataset.

This module answers three research questions about the dataset:

1. Which audio features are most associated with track popularity?
2. How do genres differ in their audio profiles?
3. Which tracks are unusual for their genre?

The core technique is enriching track-level data with genre-level
statistics using pd.merge(), then computing z-scores to measure how
much each track deviates from its genre's norms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import load_data

AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]


# ---------------------------------------------------------------------------
# Genre enrichment (merge + concat)
# ---------------------------------------------------------------------------


def enrich_with_genre_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Merge genre-level audio feature statistics onto each track.

    Computes the mean and standard deviation of each audio feature
    per genre, then merges those columns back onto the track-level
    DataFrame so every row knows its genre's average profile.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset with ``track_genre`` and audio feature columns.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with additional ``*_genre_mean`` and
        ``*_genre_std`` columns for each audio feature.
    """

    genre_means = df.groupby("track_genre")[AUDIO_FEATURES].mean()
    genre_means.columns = [f"{c}_genre_mean" for c in AUDIO_FEATURES]

    genre_stds = df.groupby("track_genre")[AUDIO_FEATURES].std()
    genre_stds.columns = [f"{c}_genre_std" for c in AUDIO_FEATURES]

    # Combine the two summary tables side by side
    genre_stats = pd.concat([genre_means, genre_stds], axis=1)

    # Merge genre-level stats back onto each track row
    enriched = pd.merge(
        df, genre_stats, left_on="track_genre", right_index=True, how="left"
    )

    return enriched


# ---------------------------------------------------------------------------
# Z-score deviation analysis
# ---------------------------------------------------------------------------


def compute_genre_deviations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute how much each track deviates from its genre's average.

    For each audio feature, a z-score is computed::

        z = (track_value - genre_mean) / genre_std

    An overall ``genre_deviation_score`` is the mean of absolute z-scores
    across all features --- a single number for how "unusual" a track is
    within its genre.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset with ``track_genre`` and audio feature columns.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with per-feature z-score columns and an
        overall ``genre_deviation_score``.
    """

    enriched = enrich_with_genre_stats(df)

    # Build an (n_tracks x n_features) z-score matrix using numpy
    z_scores = np.empty((len(enriched), len(AUDIO_FEATURES)))

    for i, feature in enumerate(AUDIO_FEATURES):
        values = enriched[feature].values.astype(float)
        means = enriched[f"{feature}_genre_mean"].values.astype(float)
        stds = enriched[f"{feature}_genre_std"].values.astype(float)

        # Replace zero/NaN stds to avoid division errors for single-track genres
        safe_stds = np.where(stds > 0, stds, np.nan)
        z_scores[:, i] = (values - means) / safe_stds

    # Overall deviation = mean absolute z-score across all features
    enriched["genre_deviation_score"] = np.round(
        np.nanmean(np.abs(z_scores), axis=1), 4
    )

    for i, feature in enumerate(AUDIO_FEATURES):
        enriched[f"{feature}_zscore"] = np.round(z_scores[:, i], 4)

    return enriched


def find_genre_outliers(
    df: pd.DataFrame,
    threshold: float = 2.0,
    top_n: int = 20,
) -> pd.DataFrame:
    """Find tracks that deviate most from their genre's typical profile.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset.
    threshold : float, default 2.0
        Minimum mean absolute z-score to count as an outlier.
    top_n : int, default 20
        Maximum number of outliers to return.

    Returns
    -------
    pd.DataFrame
        Top outlier tracks sorted by deviation score descending.
    """

    enriched = compute_genre_deviations(df)

    outliers = enriched[enriched["genre_deviation_score"] > threshold].copy()
    outliers = outliers.sort_values("genre_deviation_score", ascending=False)

    output_cols = [
        "track_id",
        "track_name",
        "artists",
        "track_genre",
        "popularity",
        "genre_deviation_score",
    ]

    return outliers[output_cols].head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature-popularity correlations
# ---------------------------------------------------------------------------


def analyze_popularity_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson correlations between each audio feature and popularity.

    Uses ``np.corrcoef`` and classifies each correlation by strength.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset with ``popularity`` and audio feature columns.

    Returns
    -------
    pd.DataFrame
        One row per feature with correlation value, absolute value,
        direction, and strength label.  Sorted by absolute correlation
        descending.
    """

    popularity = df["popularity"].values.astype(float)
    results = []

    for feature in AUDIO_FEATURES:
        feature_vals = df[feature].values.astype(float)

        # Drop rows where either value is NaN
        valid = ~(np.isnan(popularity) | np.isnan(feature_vals))
        if np.sum(valid) < 2:
            continue

        corr = np.corrcoef(popularity[valid], feature_vals[valid])[0, 1]
        abs_corr = float(np.abs(corr))

        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"

        results.append({
            "feature": feature,
            "correlation": round(float(corr), 4),
            "abs_correlation": round(abs_corr, 4),
            "direction": "positive" if corr > 0 else "negative",
            "strength": strength,
        })

    result_df = pd.DataFrame(results)
    return result_df.sort_values(
        "abs_correlation", ascending=False
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Genre comparison
# ---------------------------------------------------------------------------


def compare_genres(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Compare the most common genres by average audio profile.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset.
    top_n : int, default 10
        Number of top genres (by track count) to include.

    Returns
    -------
    pd.DataFrame
        One row per genre with track count, average popularity, and
        mean values for each audio feature.
    """

    top_genres = df["track_genre"].value_counts().head(top_n).index.tolist()
    subset = df[df["track_genre"].isin(top_genres)]

    rows = []
    for genre in top_genres:
        genre_data = subset[subset["track_genre"] == genre]
        feature_matrix = genre_data[AUDIO_FEATURES].values.astype(float)
        pop_values = genre_data["popularity"].values.astype(float)

        row = {
            "genre": genre,
            "track_count": len(genre_data),
            "avg_popularity": round(float(np.nanmean(pop_values)), 1),
        }

        for i, feature in enumerate(AUDIO_FEATURES):
            row[f"avg_{feature}"] = round(
                float(np.nanmean(feature_matrix[:, i])), 4
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature bucket analysis
# ---------------------------------------------------------------------------


def popularity_by_feature_buckets(
    df: pd.DataFrame,
    feature: str,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Split a feature into equal-width buckets and show avg popularity.

    This reveals non-linear relationships between a feature and
    popularity that a single correlation number might miss.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset.
    feature : str
        Audio feature column to bucket.
    n_buckets : int, default 5
        Number of equal-width bins to create.

    Returns
    -------
    pd.DataFrame
        One row per bucket with ``avg_popularity`` and ``track_count``.
    """

    tmp = df[[feature, "popularity"]].dropna().copy()
    values = tmp[feature].values.astype(float)

    # Create evenly spaced bucket edges with numpy
    edges = np.linspace(np.min(values), np.max(values), n_buckets + 1)
    labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(n_buckets)]

    tmp["bucket"] = pd.cut(
        tmp[feature], bins=edges, labels=labels, include_lowest=True
    )

    bucket_stats = (
        tmp.groupby("bucket", observed=False)
        .agg(
            avg_popularity=("popularity", "mean"),
            track_count=("popularity", "count"),
        )
        .round(2)
    )

    return bucket_stats


# ---------------------------------------------------------------------------
# Full analysis runner
# ---------------------------------------------------------------------------


def run_analysis(df: pd.DataFrame) -> dict[str, object]:
    """Run the full analysis pipeline and return structured results.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned Spotify dataset.

    Returns
    -------
    dict[str, object]
        Keys: ``correlations``, ``genre_comparison``, ``genre_outliers``,
        ``enriched_data``, ``feature_buckets``.
    """

    correlations = analyze_popularity_correlations(df)
    genre_comparison = compare_genres(df)
    genre_outliers = find_genre_outliers(df)
    enriched = enrich_with_genre_stats(df)

    # Run bucket analysis for the top 3 most correlated features
    top_features = correlations.head(3)["feature"].tolist()
    bucket_analyses = {}
    for feature in top_features:
        bucket_analyses[feature] = popularity_by_feature_buckets(df, feature)

    return {
        "correlations": correlations,
        "genre_comparison": genre_comparison,
        "genre_outliers": genre_outliers,
        "enriched_data": enriched,
        "feature_buckets": bucket_analyses,
    }
