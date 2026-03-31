"""Cleaning helpers for the Spotify dataset.

This module keeps the cleaning workflow separate from loading and validation so
each stage of the data pipeline has a single responsibility. The public
entrypoint is :func:`clean_data`, which applies each cleaning step in a fixed,
reproducible order and returns both the cleaned data and a summary report.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .validation import EXPECTED_COLUMNS

TEXT_COLUMNS = [
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "track_genre",
]

NUMERIC_COLUMNS = [
    "popularity",
    "duration_ms",
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "time_signature",
]

BOUNDED_AUDIO_COLUMNS = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]

REQUIRED_TEXT_COLUMNS = ["track_id", "artists", "track_name", "track_genre"]
REQUIRED_NUMERIC_COLUMNS = [
    "duration_ms",
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

BOOLEAN_TRUE_VALUES = {"true", "t", "1", "yes"}
BOOLEAN_FALSE_VALUES = {"false", "f", "0", "no"}


def validate_cleaning_columns(df: pd.DataFrame) -> None:
    """Ensure the DataFrame includes the columns needed for cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        Input Spotify dataset to clean.

    Raises
    ------
    ValueError
        If one or more expected project columns are missing.
    """

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "DataFrame is missing required columns for cleaning: "
            f"{sorted(missing)}"
        )


def drop_index_column(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop the CSV export index column when it is present.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame that may contain an ``Unnamed: 0`` column.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Cleaned DataFrame and the number of index-like columns removed.
    """

    if "Unnamed: 0" not in df.columns:
        return df, 0

    return df.drop(columns=["Unnamed: 0"]), 1


def standardize_text_fields(
    df: pd.DataFrame,
    columns: Iterable[str] = TEXT_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Trim whitespace and convert blank text values to missing values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame whose string columns should be standardized.
    columns : Iterable[str], default TEXT_COLUMNS
        Text columns to trim and inspect for empty strings.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Updated DataFrame and the number of blank values found per column.
    """

    cleaned = df.copy()
    blank_counts: dict[str, int] = {}

    for column in columns:
        if column not in cleaned.columns:
            continue

        series = cleaned[column]
        non_missing = series.notna()
        standardized = series.copy()
        # Convert to string only for the non-missing slice so existing nulls
        # stay null instead of becoming the literal text "nan".
        standardized.loc[non_missing] = (
            series.loc[non_missing].astype(str).str.strip()
        )

        blank_mask = standardized.eq("")
        blank_counts[column] = int(blank_mask.sum())
        standardized.loc[blank_mask] = pd.NA
        cleaned[column] = standardized

    return cleaned, blank_counts


def normalize_explicit_column(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Normalize the ``explicit`` flag to pandas' nullable boolean dtype.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the Spotify ``explicit`` column.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Updated DataFrame and the number of non-null values that could not be
        interpreted as booleans.
    """

    cleaned = df.copy()
    if "explicit" not in cleaned.columns:
        return cleaned, 0

    original = cleaned["explicit"].copy()
    normalized = original.astype("string").str.strip().str.lower()
    # Accept several common string encodings so the cleaning step is resilient
    # to CSV exports that store booleans in text form.
    mapped = normalized.map(
        {
            **{value: True for value in BOOLEAN_TRUE_VALUES},
            **{value: False for value in BOOLEAN_FALSE_VALUES},
        }
    )

    bool_mask = original.isin([True, False])
    mapped.loc[bool_mask] = original.loc[bool_mask].astype(bool)
    cleaned["explicit"] = mapped.astype("boolean")

    coerced_missing = int(original.notna().sum() - cleaned["explicit"].notna().sum())
    return cleaned, coerced_missing


def coerce_numeric_columns(
    df: pd.DataFrame,
    columns: Iterable[str] = NUMERIC_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Coerce numeric analysis columns and count new missing values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the project's numeric analysis columns.
    columns : Iterable[str], default NUMERIC_COLUMNS
        Numeric columns to convert with ``pandas.to_numeric``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Updated DataFrame and the number of values coerced to missing per
        numeric column.
    """

    cleaned = df.copy()
    coercion_counts: dict[str, int] = {}

    for column in columns:
        if column not in cleaned.columns:
            continue

        before_missing = cleaned[column].isna().sum()
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
        after_missing = cleaned[column].isna().sum()
        coercion_counts[column] = int(after_missing - before_missing)

    return cleaned, coercion_counts


def handle_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows missing the core identifiers needed for downstream analysis.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after text standardization and type coercion.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Updated DataFrame and the pre-drop missing-value counts for the
        required text columns.
    """

    cleaned = df.copy()
    missing_counts = {
        column: int(cleaned[column].isna().sum()) for column in REQUIRED_TEXT_COLUMNS
    }
    cleaned = cleaned.dropna(subset=REQUIRED_TEXT_COLUMNS).reset_index(drop=True)
    return cleaned, missing_counts


def remove_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove rows with invalid numeric values for the analysis workflow.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after missing text-based identifiers have been removed.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Updated DataFrame and counts of invalid numeric values detected during
        the cleaning step.
    """

    cleaned = df.copy()
    invalid_counts: dict[str, int] = {}

    for column in BOUNDED_AUDIO_COLUMNS:
        invalid_mask = cleaned[column].notna() & ~cleaned[column].between(0, 1)
        invalid_counts[column] = int(invalid_mask.sum())
        cleaned.loc[invalid_mask, column] = pd.NA

    duration_invalid = cleaned["duration_ms"].notna() & (cleaned["duration_ms"] <= 0)
    tempo_invalid = cleaned["tempo"].notna() & (cleaned["tempo"] <= 0)
    popularity_invalid = cleaned["popularity"].notna() & ~cleaned["popularity"].between(
        0, 100
    )

    invalid_counts["duration_ms"] = int(duration_invalid.sum())
    invalid_counts["tempo"] = int(tempo_invalid.sum())
    invalid_counts["popularity"] = int(popularity_invalid.sum())

    cleaned.loc[duration_invalid, "duration_ms"] = pd.NA
    cleaned.loc[tempo_invalid, "tempo"] = pd.NA
    cleaned.loc[popularity_invalid, "popularity"] = pd.NA

    # After invalid values are converted to missing, remove rows that still
    # cannot support the core feature-based analysis workflow.
    cleaned = cleaned.dropna(subset=REQUIRED_NUMERIC_COLUMNS).reset_index(drop=True)
    return cleaned, invalid_counts


def deduplicate_tracks(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop exact duplicates and keep one canonical row per ``track_id``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after text, type, and numeric cleaning.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        Deduplicated DataFrame and counts describing the duplicate removal.
    """

    cleaned = df.copy()
    exact_duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()

    original_order = pd.Series(range(len(cleaned)), index=cleaned.index)
    completeness = cleaned.notna().sum(axis=1)
    popularity = cleaned["popularity"].fillna(-1)

    # Rank repeated track IDs so the most complete row survives. Popularity is
    # used as a secondary tie-breaker, then the original order keeps the
    # result deterministic when rows are otherwise identical in quality.
    ranked = cleaned.assign(
        _completeness=completeness,
        _popularity_rank=popularity,
        _original_order=original_order,
    )

    ranked = ranked.sort_values(
        by=["track_id", "_completeness", "_popularity_rank", "_original_order"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )

    duplicate_track_ids = int(ranked["track_id"].duplicated(keep="first").sum())
    cleaned = (
        ranked.drop_duplicates(subset="track_id", keep="first")
        .drop(columns=["_completeness", "_popularity_rank", "_original_order"])
        .reset_index(drop=True)
    )

    return cleaned, {
        "duplicate_rows_removed": exact_duplicates,
        "duplicate_track_ids_removed": duplicate_track_ids,
    }


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run the full data-cleaning pipeline and return a summary report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Spotify dataset to clean.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, object]]
        Cleaned DataFrame plus a report describing how many values or rows were
        modified or removed at each cleaning stage.
    """

    validate_cleaning_columns(df)

    # Work on a copy so callers keep access to the original raw DataFrame.
    cleaned = df.copy()
    initial_rows = int(len(cleaned))

    cleaned, dropped_index_columns = drop_index_column(cleaned)
    cleaned, blank_text_counts = standardize_text_fields(cleaned)
    cleaned, explicit_coercions = normalize_explicit_column(cleaned)
    cleaned, numeric_coercions = coerce_numeric_columns(cleaned)

    before_missing_drop = len(cleaned)
    cleaned, missing_counts = handle_missing_values(cleaned)
    rows_removed_for_missing = int(before_missing_drop - len(cleaned))

    before_invalid_drop = len(cleaned)
    cleaned, invalid_counts = remove_invalid_rows(cleaned)
    rows_removed_for_invalid = int(before_invalid_drop - len(cleaned))

    before_deduplicate = len(cleaned)
    cleaned, duplicate_summary = deduplicate_tracks(cleaned)
    rows_removed_for_duplicates = int(before_deduplicate - len(cleaned))

    report = {
        "input_rows": initial_rows,
        "output_rows": int(len(cleaned)),
        "rows_removed_total": int(initial_rows - len(cleaned)),
        "index_columns_removed": dropped_index_columns,
        "blank_text_values": blank_text_counts,
        "explicit_values_coerced_to_missing": explicit_coercions,
        "numeric_values_coerced_to_missing": numeric_coercions,
        "missing_values_before_drop": missing_counts,
        "rows_removed_for_missing_values": rows_removed_for_missing,
        "invalid_values_detected": invalid_counts,
        "rows_removed_for_invalid_values": rows_removed_for_invalid,
        "rows_removed_for_duplicates": rows_removed_for_duplicates,
        **duplicate_summary,
    }

    return cleaned.reset_index(drop=True), report


def run_cleaning(path: str) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load the raw dataset from disk and return the cleaned result.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, object]]
        Cleaned DataFrame and the cleaning summary report.
    """

    from .io import load_data

    df = load_data(path)
    return clean_data(df)
