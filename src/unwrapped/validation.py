"""Validation helpers for the Spotify dataset."""

import pandas as pd

EXPECTED_COLUMNS = {
    "track_id",
    "artists",
    "album_name",
    "track_name",
    "popularity",
    "duration_ms",
    "explicit",
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
    "track_genre",
}


def validate_schema(df: pd.DataFrame) -> None:
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    extra = set(df.columns) - EXPECTED_COLUMNS
    if extra:
        print(f"[Warning] Extra columns detected: {extra}")


_BOUNDED_UNIT_COLUMNS = (
    "danceability",
    "energy",
    "valence",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "liveness"
)

_POSITIVE_COLUMNS = ("tempo", "duration_ms")


def validate_ranges(df: pd.DataFrame) -> None:
    errors = []

    def check_range(col: str, low: float, high: float) -> None:
        if not df[col].dropna().between(low, high).all():
            errors.append(f"{col} out of range [{low}, {high}]")

    for col in _BOUNDED_UNIT_COLUMNS:
        check_range(col, 0, 1)

    if (df["tempo"] <= 0).any():
        errors.append("tempo contains non-positive values")

    if (df["duration_ms"] <= 0).any():
        errors.append("duration_ms contains non-positive values")

    if errors:
        raise ValueError("Range validation failed:\n" + "\n".join(errors))


def range_violation_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count out-of-range values per column without raising.

    Parallels :func:`validate_ranges` but returns per-column counts so the
    validation entrypoint can report on real-world datasets that contain
    a handful of dirty rows (e.g. ``tempo == 0``) without aborting.
    """
    counts: dict[str, int] = {}

    for col in _BOUNDED_UNIT_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        counts[col] = int((~series.between(0, 1)).sum())

    for col in _POSITIVE_COLUMNS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        counts[col] = int((series <= 0).sum())

    return counts


def validate_duplicates(df: pd.DataFrame) -> dict:
    duplicate_rows = df.duplicated().sum()
    duplicate_tracks = df["track_id"].duplicated().sum()

    return {
        "duplicate_rows": int(duplicate_rows),
        "duplicate_track_ids": int(duplicate_tracks),
    }


def validate_track_consistency(df: pd.DataFrame) -> int:
    grouped = df.groupby("track_id")[
        ["danceability", "energy", "tempo", "duration_ms"]
    ].nunique()

    inconsistent_tracks = grouped[(grouped > 1).any(axis=1)]

    return int(len(inconsistent_tracks))


def validate_correlations(df: pd.DataFrame) -> None:
    corr = df[["energy", "loudness"]].corr().iloc[0, 1]

    if corr < 0.5:
        print(f"[Warning] Weak correlation between energy and loudness: {corr:.2f}")


def missing_summary(df: pd.DataFrame) -> dict:
    return df.isnull().sum().to_dict()


def validation_report(df: pd.DataFrame) -> dict:
    report = {}

    report["num_rows"] = int(len(df))
    report["num_columns"] = int(len(df.columns))
    report["missing_values"] = missing_summary(df)
    report.update(validate_duplicates(df))
    report["unique_tracks"] = int(df["track_id"].nunique())
    report["inconsistent_tracks"] = validate_track_consistency(df)
    report["range_violations"] = range_violation_counts(df)

    return report


def run_validation(path: str):
    from .io import load_data

    df = load_data(path)

    validate_schema(df)
    validate_correlations(df)

    report = validation_report(df)

    return df, report
