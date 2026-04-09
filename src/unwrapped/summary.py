import pandas as pd


def summarize_dataset(df):
    """
    Return a basic summary of a Spotify dataset.
    """
    summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
    }

    if "artist_name" in df.columns:
        summary["unique_artists"] = df["artist_name"].nunique()

    if "genre" in df.columns:
        summary["unique_genres"] = df["genre"].nunique()
        summary["top_genres"] = df["genre"].value_counts().head(10).to_dict()

    if "track_name" in df.columns:
        summary["unique_tracks"] = df["track_name"].nunique()

    if "popularity" in df.columns:
        summary["mean_popularity"] = df["popularity"].mean()
        summary["median_popularity"] = df["popularity"].median()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        summary["numeric_summary"] = df[numeric_cols].describe().to_dict()

    return summary