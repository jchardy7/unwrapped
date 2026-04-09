"""Descriptive summary helpers for the Spotify dataset.

This module provides exploratory data analysis functions that produce a
comprehensive summary of a Spotify DataFrame.  The public entrypoint is
:func:`summarize_data`, which calls every helper and returns a combined report
dictionary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .clean import BOUNDED_AUDIO_COLUMNS, NUMERIC_COLUMNS, TEXT_COLUMNS

AUDIO_FEATURE_COLUMNS = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "loudness",
    "tempo",
    "duration_ms",
]

CATEGORICAL_COLUMNS = ["track_genre", "artists", "explicit"]


def describe_shape(df: pd.DataFrame) -> dict[str, object]:
    """Return basic shape information and dtype breakdown.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset to describe.

    Returns
    -------
    dict[str, object]
        Row count, column count, and per-dtype column counts.
    """

    dtype_counts = df.dtypes.apply(lambda d: str(d)).value_counts().to_dict()

    return {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": list(df.columns),
        "dtype_counts": dtype_counts,
    }


def describe_numeric(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute descriptive statistics for each numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset to describe.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of column name to its summary statistics including count,
        mean, std, min, Q1, median, Q3, max, skewness, and kurtosis.
    """

    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    result: dict[str, dict[str, float]] = {}

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        result[col] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "q1": float(series.quantile(0.25)),
            "median": float(series.median()),
            "q3": float(series.quantile(0.75)),
            "max": float(series.max()),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
        }

    return result


def describe_categorical(
    df: pd.DataFrame,
    top_n: int = 10,
) -> dict[str, dict[str, object]]:
    """Summarize categorical columns with unique counts and top values.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset to describe.
    top_n : int, default 10
        Number of most-frequent values to include per column.

    Returns
    -------
    dict[str, dict[str, object]]
        Mapping of column name to unique count and top-N value frequencies.
    """

    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    result: dict[str, dict[str, object]] = {}

    for col in cat_cols:
        series = df[col].dropna()
        top_values = series.value_counts().head(top_n)
        result[col] = {
            "num_unique": int(series.nunique()),
            "top_values": {str(k): int(v) for k, v in top_values.items()},
        }

    return result


def describe_missing(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Report missing-value counts and percentages per column.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset to describe.

    Returns
    -------
    dict[str, dict[str, object]]
        Mapping of column name to missing count and percentage.
    """

    total = len(df)
    result: dict[str, dict[str, object]] = {}

    for col in df.columns:
        missing = int(df[col].isna().sum())
        result[col] = {
            "count": missing,
            "percentage": round(missing / total * 100, 2) if total > 0 else 0.0,
        }

    return result


def detect_outliers(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Flag outliers using the 1.5x IQR method for each numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset to inspect.

    Returns
    -------
    dict[str, dict[str, object]]
        Mapping of column name to outlier count, percentage, and bounds.
    """

    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    result: dict[str, dict[str, object]] = {}

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_mask = (series < lower) | (series > upper)
        count = int(outlier_mask.sum())

        result[col] = {
            "count": count,
            "percentage": round(count / len(series) * 100, 2),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
        }

    return result


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise Pearson correlations for audio feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset containing audio feature columns.

    Returns
    -------
    pd.DataFrame
        Correlation matrix for the available audio feature columns.
    """

    cols = [c for c in AUDIO_FEATURE_COLUMNS if c in df.columns]
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    return numeric.corr()


def target_correlations(
    df: pd.DataFrame,
    target: str = "popularity",
) -> dict[str, float]:
    """Compute each audio feature's Pearson correlation with the target.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset containing audio features and the target column.
    target : str, default "popularity"
        Column to correlate against.

    Returns
    -------
    dict[str, float]
        Mapping of feature name to its correlation with the target, sorted
        by absolute value descending.
    """

    if target not in df.columns:
        return {}

    target_series = pd.to_numeric(df[target], errors="coerce")
    correlations: dict[str, float] = {}

    for col in AUDIO_FEATURE_COLUMNS:
        if col not in df.columns or col == target:
            continue
        feature = pd.to_numeric(df[col], errors="coerce")
        valid = target_series.notna() & feature.notna()
        if valid.sum() < 2:
            continue
        correlations[col] = round(float(target_series[valid].corr(feature[valid])), 4)

    return dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))


def genre_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate audio features by genre with count, mean, and std.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset containing ``track_genre`` and audio features.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with genre as index and (feature, stat) columns.
    """

    if "track_genre" not in df.columns:
        return pd.DataFrame()

    feature_cols = [c for c in AUDIO_FEATURE_COLUMNS if c in df.columns]
    numeric = df[["track_genre"] + feature_cols].copy()
    for col in feature_cols:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")

    grouped = numeric.groupby("track_genre")[feature_cols].agg(["count", "mean", "std"])
    return grouped


def popularity_by_genre_pivot(
    df: pd.DataFrame,
    bins: list[int] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Pivot table of track counts by genre and popularity tier.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset containing ``track_genre`` and ``popularity``.
    bins : list[int] | None
        Bin edges for popularity. Defaults to ``[0, 25, 50, 75, 100]``.
    labels : list[str] | None
        Labels for the bins. Defaults to ``["Low", "Medium", "High", "Very High"]``.

    Returns
    -------
    pd.DataFrame
        Pivot table with genres as rows, popularity tiers as columns, and
        track counts as values.
    """

    if "track_genre" not in df.columns or "popularity" not in df.columns:
        return pd.DataFrame()

    if bins is None:
        bins = [0, 25, 50, 75, 100]
    if labels is None:
        labels = ["Low", "Medium", "High", "Very High"]

    tmp = df[["track_genre", "popularity"]].copy()
    tmp["popularity"] = pd.to_numeric(tmp["popularity"], errors="coerce")
    tmp = tmp.dropna(subset=["popularity"])
    tmp["popularity_tier"] = pd.cut(tmp["popularity"], bins=bins, labels=labels, include_lowest=True)

    return pd.pivot_table(
        tmp,
        index="track_genre",
        columns="popularity_tier",
        aggfunc="size",
        fill_value=0,
        observed=False,
    )


def summarize_data(df: pd.DataFrame) -> dict[str, object]:
    """Run the full descriptive summary and return a combined report.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset to summarize.

    Returns
    -------
    dict[str, object]
        Combined report containing shape, numeric stats, categorical stats,
        missing values, outliers, correlations, target correlations, and
        genre-level aggregations.
    """

    corr = correlation_matrix(df)
    genre = genre_summary(df)
    pivot = popularity_by_genre_pivot(df)

    return {
        "shape": describe_shape(df),
        "numeric": describe_numeric(df),
        "categorical": describe_categorical(df),
        "missing": describe_missing(df),
        "outliers": detect_outliers(df),
        "correlation_matrix": corr.to_dict(),
        "target_correlations": target_correlations(df),
        "genre_summary": genre.to_dict(),
        "popularity_by_genre": pivot.to_dict(),
    }


def export_summary_csvs(df: pd.DataFrame, output_dir: str = "outputs") -> dict[str, str]:
    """Export key summary results as CSV files for the visualization module.

    Writes four CSV files that the visualization team can load directly
    for plotting:

    - ``summary_target_correlations.csv`` — each audio feature's Pearson
      correlation with popularity, sorted by absolute strength.
    - ``summary_genre_means.csv`` — mean audio features per genre with
      track counts, derived from the genre aggregation.
    - ``summary_popularity_by_genre.csv`` — pivot table of track counts
      across popularity tiers (Low / Medium / High / Very High) per genre.
    - ``summary_outliers.csv`` — per-column outlier counts, percentages,
      and IQR bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Spotify dataset to summarize and export.
    output_dir : str, default "outputs"
        Directory to write CSV files into. Created if it does not exist.

    Returns
    -------
    dict[str, str]
        Mapping of description to file path for each exported CSV.
    """

    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exported: dict[str, str] = {}

    # Target correlations
    corrs = target_correlations(df)
    if corrs:
        corr_df = pd.DataFrame(
            list(corrs.items()), columns=["feature", "correlation"]
        )
        path = str(out / "summary_target_correlations.csv")
        corr_df.to_csv(path, index=False)
        exported["target_correlations"] = path

    # Genre means with counts
    genre = genre_summary(df)
    if not genre.empty:
        # Flatten the MultiIndex columns into "feature_stat" format
        genre.columns = [f"{feat}_{stat}" for feat, stat in genre.columns]
        path = str(out / "summary_genre_means.csv")
        genre.to_csv(path)
        exported["genre_means"] = path

    # Popularity by genre pivot
    pivot = popularity_by_genre_pivot(df)
    if not pivot.empty:
        path = str(out / "summary_popularity_by_genre.csv")
        pivot.to_csv(path)
        exported["popularity_by_genre"] = path

    # Outlier summary
    outliers = detect_outliers(df)
    if outliers:
        outlier_df = pd.DataFrame(outliers).T
        outlier_df.index.name = "column"
        path = str(out / "summary_outliers.csv")
        outlier_df.to_csv(path)
        exported["outliers"] = path

    return exported


def run_summary(path: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the dataset from disk and return the descriptive summary.

    Parameters
    ----------
    path : str
        Path to the CSV file to summarize.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, object]]
        Loaded DataFrame and the summary report.
    """

    from .io import load_data

    df = load_data(path)
    return df, summarize_data(df)
