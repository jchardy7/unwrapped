"""Unsupervised clustering of Spotify tracks by audio profile.

Provides utilities to scale features, search for a good ``k`` (elbow +
silhouette), fit a ``KMeans`` model, recover centroids in original feature
units, and summarize each cluster.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .constants import AUDIO_FEATURES
from .io import load_data


SILHOUETTE_SAMPLE_SIZE = 2000


def prepare_clustering_data(
    df: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[np.ndarray, StandardScaler, list[str], pd.Index]:
    """Drop NA rows on the chosen features and standard-scale them.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    features : list[str], optional
        Columns to cluster on. Defaults to :data:`AUDIO_FEATURES`.

    Returns
    -------
    tuple
        ``(X_scaled, scaler, feature_names, kept_index)`` — the kept index
        lets callers re-attach cluster labels to the original frame.
    """
    if features is None:
        features = list(AUDIO_FEATURES)

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    subset = df[features].apply(pd.to_numeric, errors="coerce").dropna()
    if subset.empty:
        raise ValueError("No rows with non-null values for the chosen features.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(subset.to_numpy())

    return X_scaled, scaler, features, subset.index


def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: Iterable[int] = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """Sweep ``k`` and report inertia (elbow) and silhouette per value.

    Silhouette is computed on a 2000-row sample for speed when
    ``len(X_scaled) > SILHOUETTE_SAMPLE_SIZE``.
    """
    rows = []
    rng = np.random.default_rng(random_state)
    n = len(X_scaled)
    sample_idx = (
        rng.choice(n, size=SILHOUETTE_SAMPLE_SIZE, replace=False)
        if n > SILHOUETTE_SAMPLE_SIZE
        else np.arange(n)
    )

    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = model.fit_predict(X_scaled)
        sample_labels = labels[sample_idx]
        if len(np.unique(sample_labels)) < 2:
            silhouette = float("nan")
        else:
            silhouette = float(silhouette_score(X_scaled[sample_idx], sample_labels))

        rows.append(
            {
                "k": int(k),
                "inertia": float(model.inertia_),
                "silhouette": silhouette,
            }
        )

    return pd.DataFrame(rows)


def cluster_songs(
    df: pd.DataFrame,
    n_clusters: int = 5,
    features: list[str] | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Fit a KMeans model and attach a ``cluster`` column to ``df``.

    Rows missing any of the chosen features get ``cluster = -1`` so the
    returned frame keeps the same row count as the input.
    """
    X_scaled, scaler, feature_names, kept_index = prepare_clustering_data(
        df, features=features
    )

    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(X_scaled)

    out = df.copy()
    out["cluster"] = -1
    out.loc[kept_index, "cluster"] = labels
    return out, model, scaler


def cluster_centroids(
    model: KMeans, scaler: StandardScaler, feature_names: list[str]
) -> pd.DataFrame:
    """Return centroids in original (unscaled) feature units."""
    centroids = scaler.inverse_transform(model.cluster_centers_)
    return pd.DataFrame(
        centroids,
        columns=feature_names,
        index=pd.Index(range(len(centroids)), name="cluster"),
    )


def cluster_summary(
    df_with_clusters: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Per-cluster size, mean popularity, mean of each feature, and top genre."""
    if features is None:
        features = list(AUDIO_FEATURES)

    clustered = df_with_clusters[df_with_clusters["cluster"] >= 0]

    rows = []
    for cluster_id, group in clustered.groupby("cluster"):
        row: dict[str, Any] = {
            "cluster": int(cluster_id),
            "size": len(group),
        }
        if "popularity" in group.columns:
            row["mean_popularity"] = float(group["popularity"].mean())
        for feat in features:
            if feat in group.columns:
                row[f"mean_{feat}"] = float(group[feat].mean())
        if "track_genre" in group.columns and not group["track_genre"].dropna().empty:
            row["top_genre"] = group["track_genre"].mode().iloc[0]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def save_outputs(
    assignments_df: pd.DataFrame,
    centroids_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    k_search_df: pd.DataFrame | None,
    output_dir: str | Path = "outputs",
) -> dict[str, str]:
    """Write clustering artifacts to ``output_dir``."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "assignments": output_path / "cluster_assignments.csv",
        "centroids": output_path / "cluster_centroids.csv",
        "summary": output_path / "cluster_summary.csv",
    }

    assignments_df.to_csv(paths["assignments"], index=False)
    centroids_df.to_csv(paths["centroids"])
    summary_df.to_csv(paths["summary"], index=False)

    if k_search_df is not None:
        k_path = output_path / "cluster_k_search.csv"
        k_search_df.to_csv(k_path, index=False)
        paths["k_search"] = k_path

    return {k: str(v) for k, v in paths.items()}


def run_clustering_pipeline(
    data_path: str = "data/raw/spotify_data.csv",
    n_clusters: int = 5,
    k_search_range: Iterable[int] | None = range(2, 11),
    save_results: bool = True,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    """End-to-end pipeline: load, prep, search k (optional), cluster, summarize."""
    df = load_data(data_path)

    X_scaled, scaler, feature_names, kept_index = prepare_clustering_data(df)

    k_search_df = (
        find_optimal_k(X_scaled, k_range=k_search_range)
        if k_search_range is not None
        else None
    )

    df_with_clusters, model, _ = cluster_songs(df, n_clusters=n_clusters)
    centroids_df = cluster_centroids(model, scaler, feature_names)
    summary_df = cluster_summary(df_with_clusters)

    assignments_cols = [c for c in ("track_id", "track_name", "artists", "track_genre") if c in df_with_clusters.columns]
    assignments_df = df_with_clusters.loc[df_with_clusters["cluster"] >= 0, assignments_cols + ["cluster"]]

    if save_results:
        save_outputs(
            assignments_df,
            centroids_df,
            summary_df,
            k_search_df,
            output_dir=output_dir,
        )

    return {
        "model": model,
        "scaler": scaler,
        "assignments": assignments_df,
        "centroids": centroids_df,
        "summary": summary_df,
        "k_search": k_search_df,
    }


def main() -> None:
    """Run the default clustering pipeline."""
    run_clustering_pipeline("data/raw/spotify_data.csv")


if __name__ == "__main__":
    main()
