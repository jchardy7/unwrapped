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
from scipy import stats

from .constants import AUDIO_FEATURES
from .io import load_data


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

    # Overall deviation = mean absolute z-score across all features.
    # Tracks in single-track genres have all-NaN z-score rows (std is 0),
    # so compute nanmean only for rows that have at least one valid value
    # and leave the rest as NaN — avoids a "Mean of empty slice" warning.
    abs_z = np.abs(z_scores)
    has_valid = ~np.all(np.isnan(abs_z), axis=1)
    deviation = np.full(len(enriched), np.nan)
    if has_valid.any():
        deviation[has_valid] = np.nanmean(abs_z[has_valid], axis=1)
    enriched["genre_deviation_score"] = np.round(deviation, 4)

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


def _bootstrap_correlation_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for Pearson r.

    Resamples (x, y) pairs with replacement ``n_bootstrap`` times and
    returns the central ``1 - alpha`` interval of the resampled r values.
    Returns ``(nan, nan)`` when ``n_bootstrap`` is zero or every resample
    degenerates (e.g. a constant feature).
    """
    if n_bootstrap <= 0:
        return float("nan"), float("nan")

    n = len(x)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        # Skip resamples where one side is constant; corrcoef would warn
        # and return NaN, which nanpercentile handles downstream.
        if xb.std() == 0 or yb.std() == 0:
            samples[i] = np.nan
        else:
            samples[i] = np.corrcoef(xb, yb)[0, 1]

    if np.all(np.isnan(samples)):
        return float("nan"), float("nan")

    lower = float(np.nanpercentile(samples, 100 * alpha / 2))
    upper = float(np.nanpercentile(samples, 100 * (1 - alpha / 2)))
    return lower, upper


def _holm_bonferroni(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment for a family of p-values.

    Returns adjusted p-values in the same order as the input. NaN inputs
    are left as NaN and excluded from the correction (they don't count
    toward the family size).
    """
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p, np.nan)
    tested_mask = ~np.isnan(p)
    tested = p[tested_mask]

    if tested.size == 0:
        return adjusted

    order = np.argsort(tested)
    ranked = tested[order]
    factors = np.arange(tested.size, 0, -1)  # m, m-1, ..., 1
    raw = ranked * factors
    monotone = np.maximum.accumulate(raw)
    capped = np.minimum(monotone, 1.0)

    back = np.empty_like(tested)
    back[order] = capped
    adjusted[tested_mask] = back
    return adjusted


def analyze_popularity_correlations(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Compute Pearson correlations between each audio feature and popularity.

    Each correlation is reported with a two-sided p-value, a percentile
    bootstrap confidence interval, a Holm–Bonferroni-adjusted p-value
    to account for testing nine features simultaneously, and a strength
    label for quick scanning.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset with ``popularity`` and audio feature columns.
    n_bootstrap : int, default 1000
        Bootstrap resamples used to build the confidence interval. Set
        to ``0`` to skip the bootstrap (``ci_low``/``ci_high`` become NaN).
    alpha : float, default 0.05
        Significance level. Controls both the CI width (``1 - alpha``)
        and the ``significant`` column after Holm adjustment.
    random_state : int or None, default 42
        Seed for the bootstrap resampler. ``None`` uses fresh entropy.

    Returns
    -------
    pd.DataFrame
        One row per feature with columns ``feature``, ``correlation``,
        ``abs_correlation``, ``direction``, ``strength``, ``ci_low``,
        ``ci_high``, ``p_value``, ``p_value_adjusted``, ``significant``.
        Sorted by absolute correlation descending.
    """

    popularity = df["popularity"].values.astype(float)
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []

    for feature in AUDIO_FEATURES:
        feature_vals = df[feature].values.astype(float)

        valid = ~(np.isnan(popularity) | np.isnan(feature_vals))
        if np.sum(valid) < 3:
            continue

        x = popularity[valid]
        y = feature_vals[valid]

        # Skip constant columns: scipy.stats.pearsonr would warn and
        # return NaN, and the bootstrap CI would be meaningless.
        if x.std() == 0 or y.std() == 0:
            continue

        pearson = stats.pearsonr(x, y)
        corr = float(pearson.statistic)
        p_value = float(pearson.pvalue)
        abs_corr = float(np.abs(corr))

        ci_low, ci_high = _bootstrap_correlation_ci(
            x, y, n_bootstrap=n_bootstrap, alpha=alpha, rng=rng
        )

        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"

        rows.append({
            "feature": feature,
            "correlation": round(corr, 4),
            "abs_correlation": round(abs_corr, 4),
            "direction": "positive" if corr > 0 else "negative",
            "strength": strength,
            "ci_low": round(ci_low, 4) if not np.isnan(ci_low) else np.nan,
            "ci_high": round(ci_high, 4) if not np.isnan(ci_high) else np.nan,
            "p_value": p_value,
            "n": int(valid.sum()),
        })

    if not rows:
        return pd.DataFrame(
            columns=[
                "feature", "correlation", "abs_correlation", "direction",
                "strength", "ci_low", "ci_high", "p_value",
                "p_value_adjusted", "significant", "n",
            ]
        )

    result_df = pd.DataFrame(rows)

    adjusted = _holm_bonferroni(result_df["p_value"].to_numpy())
    result_df["p_value_adjusted"] = adjusted
    result_df["significant"] = result_df["p_value_adjusted"] < alpha

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

    if feature not in df.columns:
        raise ValueError(
            f"feature '{feature}' not in DataFrame. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )
    if "popularity" not in df.columns:
        raise ValueError("DataFrame must contain a 'popularity' column.")
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")

    tmp = df[[feature, "popularity"]].dropna().copy()
    if tmp.empty:
        return pd.DataFrame(columns=["avg_popularity", "track_count"])

    values = tmp[feature].values.astype(float)
    value_min = float(np.min(values))
    value_max = float(np.max(values))

    if value_min == value_max:
        label = f"{value_min:.2f}-{value_max:.2f}"
        bucket_stats = pd.DataFrame(
            {
                "avg_popularity": [round(float(tmp["popularity"].mean()), 2)],
                "track_count": [int(tmp["popularity"].count())],
            },
            index=pd.CategoricalIndex([label], categories=[label], name="bucket"),
        )
        return bucket_stats

    # Create evenly spaced bucket edges with numpy
    edges = np.linspace(value_min, value_max, n_buckets + 1)
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
