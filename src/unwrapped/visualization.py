from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import AUDIO_FEATURES


def plot_top_genres(df: pd.DataFrame, top_n: int = 10):
    """
    Plot the top N track genres by count.
    """
    if "track_genre" not in df.columns:
        raise ValueError("DataFrame must contain a 'track_genre' column.")

    genre_counts = (
        df["track_genre"]
        .dropna()
        .astype(str)
        .value_counts()
        .head(top_n)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    genre_counts.plot(kind="barh", ax=ax)

    ax.set_title(f"Top {top_n} Track Genres")
    ax.set_xlabel("Number of Tracks")
    ax.set_ylabel("Genre")
    fig.tight_layout()

    return fig, ax


def plot_popularity_distribution(df: pd.DataFrame, bins: int = 20):
    """
    Plot a histogram of track popularity.
    """
    if "popularity" not in df.columns:
        raise ValueError("DataFrame must contain a 'popularity' column.")

    popularity = pd.to_numeric(df["popularity"], errors="coerce").dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(popularity, bins=bins)

    ax.set_title("Distribution of Track Popularity")
    ax.set_xlabel("Popularity")
    ax.set_ylabel("Count")
    fig.tight_layout()

    return fig, ax


def plot_feature_correlations(df: pd.DataFrame):
    """
    Horizontal bar chart of each audio feature's Pearson correlation with popularity.
    Bars are colored green (positive) or red (negative) for quick visual scanning.
    """
    required = AUDIO_FEATURES + ["popularity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    popularity = pd.to_numeric(df["popularity"], errors="coerce")
    correlations = {}
    for feature in AUDIO_FEATURES:
        values = pd.to_numeric(df[feature], errors="coerce")
        mask = popularity.notna() & values.notna()
        if mask.sum() < 2:
            correlations[feature] = 0.0
        else:
            correlations[feature] = float(np.corrcoef(values[mask], popularity[mask])[0, 1])

    corr_series = pd.Series(correlations).sort_values()

    colors = ["#e05c5c" if v < 0 else "#5cae5c" for v in corr_series]

    fig, ax = plt.subplots(figsize=(8, 5))
    corr_series.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Audio Feature Correlations with Popularity")
    ax.set_xlabel("Pearson r")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    return fig, ax


def plot_correlation_forest(
    df: pd.DataFrame,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int | None = 42,
):
    """
    Forest plot of each audio feature's Pearson correlation with popularity,
    annotated with bootstrap confidence intervals and Holm-adjusted significance.

    Features that remain significant after the Holm–Bonferroni correction are
    drawn in the series color; non-significant features are drawn in grey so
    they're visually deprioritized.
    """
    from .analysis import analyze_popularity_correlations

    corr_df = analyze_popularity_correlations(
        df,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )
    if corr_df.empty:
        raise ValueError("No features available for correlation forest plot.")

    # Draw strongest |r| at the top so the eye moves down-then-weaker.
    corr_df = corr_df.sort_values("abs_correlation", ascending=True).reset_index(
        drop=True
    )

    y_pos = np.arange(len(corr_df))
    r = corr_df["correlation"].to_numpy()
    lo = corr_df["ci_low"].to_numpy()
    hi = corr_df["ci_high"].to_numpy()
    xerr = np.vstack([r - lo, hi - r])

    significant = corr_df["significant"].fillna(False).to_numpy()
    colors = np.where(significant, "#1DB954", "#b3b3b3")

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(len(corr_df)):
        ax.errorbar(
            r[i],
            y_pos[i],
            xerr=np.array([[xerr[0, i]], [xerr[1, i]]]),
            fmt="o",
            color=colors[i],
            ecolor=colors[i],
            capsize=4,
            markersize=6,
            linewidth=1.5,
        )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(corr_df["feature"])
    ax.set_xlabel("Pearson r with popularity (bootstrap 95% CI)")
    ax.set_title(
        f"Audio Feature Correlations with Popularity "
        f"(Holm-adjusted α = {alpha:g})"
    )

    # Legend explaining the color split without relying on a plot artist.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="#1DB954", linestyle="",
               label="significant"),
        Line2D([0], [0], marker="o", color="#b3b3b3", linestyle="",
               label="not significant"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.tight_layout()

    return fig, ax


def plot_genre_popularity(df: pd.DataFrame, top_n: int = 15):
    """
    Bar chart of mean popularity score for the top N genres by track count.
    Error bars show one standard deviation.
    """
    if "track_genre" not in df.columns or "popularity" not in df.columns:
        raise ValueError("DataFrame must contain 'track_genre' and 'popularity' columns.")

    pop = pd.to_numeric(df["popularity"], errors="coerce")
    tmp = df.assign(popularity=pop).dropna(subset=["popularity", "track_genre"])

    counts = tmp["track_genre"].value_counts()
    top_genres = counts.head(top_n).index

    grouped = tmp[tmp["track_genre"].isin(top_genres)].groupby("track_genre")["popularity"]
    means = grouped.mean().reindex(top_genres).sort_values(ascending=False)
    stds = grouped.std().reindex(means.index)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(means))
    ax.bar(x, means.values, yerr=stds.values, capsize=4, color="#1DB954", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(means.index, rotation=45, ha="right")
    ax.set_title(f"Mean Popularity by Genre (Top {top_n} Genres by Track Count)")
    ax.set_ylabel("Mean Popularity")
    ax.set_xlabel("Genre")
    fig.tight_layout()

    return fig, ax


def plot_audio_heatmap(df: pd.DataFrame):
    """
    Correlation heatmap of all audio features using annotated squares.
    """
    missing = [c for c in AUDIO_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    numeric = df[AUDIO_FEATURES].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    n = len(AUDIO_FEATURES)
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdYlGn", aspect="auto")
    fig.colorbar(im, ax=ax, label="Pearson r")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(AUDIO_FEATURES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(AUDIO_FEATURES, fontsize=9)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)

    ax.set_title("Audio Feature Correlation Heatmap")
    fig.tight_layout()

    return fig, ax


def plot_hit_vs_nonhit_profiles(df: pd.DataFrame, threshold: int = 70):
    """
    Grouped bar chart comparing mean audio feature values for hits vs non-hits.
    A track is a hit if its popularity is >= threshold.
    Features are min-max scaled to [0, 1] so all bars are on the same axis.
    """
    required = AUDIO_FEATURES + ["popularity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")

    tmp = df[required].copy()
    for col in AUDIO_FEATURES:
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp["popularity"] = pd.to_numeric(tmp["popularity"], errors="coerce")
    tmp = tmp.dropna(subset=["popularity"])
    tmp["is_hit"] = tmp["popularity"] >= threshold

    for col in AUDIO_FEATURES:
        col_min = tmp[col].min()
        col_max = tmp[col].max()
        rng = col_max - col_min
        if rng > 0:
            tmp[col] = (tmp[col] - col_min) / rng

    hit_means = tmp[tmp["is_hit"]][AUDIO_FEATURES].mean()
    nonhit_means = tmp[~tmp["is_hit"]][AUDIO_FEATURES].mean()

    x = np.arange(len(AUDIO_FEATURES))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, hit_means.values, width, label=f"Hits (popularity ≥ {threshold})", color="#1DB954")
    ax.bar(x + width / 2, nonhit_means.values, width, label="Non-hits", color="#b3b3b3")

    ax.set_xticks(x)
    ax.set_xticklabels(AUDIO_FEATURES, rotation=45, ha="right")
    ax.set_ylabel("Mean Value (scaled 0–1)")
    ax.set_title("Audio Feature Profiles: Hits vs Non-Hits")
    ax.legend()
    fig.tight_layout()

    return fig, ax

def plot_feature_scatter(
    df: pd.DataFrame,
    x_feature: str = "energy",
    y_feature: str = "danceability",
    sample: int = 2000,
):

    for col in [x_feature, y_feature, "popularity"]:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column.")
 
    plot_df = df[[x_feature, y_feature, "popularity"]].copy()
    for col in [x_feature, y_feature, "popularity"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
    plot_df = plot_df.dropna()
 
    if len(plot_df) > sample:
        plot_df = plot_df.sample(n=sample, random_state=42)
 
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(
        plot_df[x_feature],
        plot_df[y_feature],
        c=plot_df["popularity"],
        cmap="viridis",
        alpha=0.5,
        s=12,
    )
    fig.colorbar(sc, ax=ax, label="Popularity")
    ax.set_xlabel(x_feature.capitalize())
    ax.set_ylabel(y_feature.capitalize())
    ax.set_title(f"{x_feature.capitalize()} vs {y_feature.capitalize()} (colored by Popularity)")
    fig.tight_layout()
 
    return fig, ax

def save_figure(fig, output_path: str | Path) -> None:
    """
    Save a matplotlib figure to disk, creating parent directories if needed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")


def plot_tempo_by_genre(df: pd.DataFrame, top_n: int = 10):
    """
    Box plot of tempo distribution across the top N genres by track count.
 
    Uses GroupBy aggregation to pull tempo values per genre, giving a clear
    picture of which genres are fast- vs slow-paced.
 
    Parameters
    ----------
    df : pd.DataFrame
        Cleaned Spotify tracks DataFrame.
    top_n : int
        Number of top genres (by track count) to include. Defaults to 10.
def plot_preference_scores(scores_df: pd.DataFrame, top_n: int = 20):
    """
    Horizontal bar chart of the top N tracks by preference score.
 
    Designed to work directly with the DataFrame returned by
    ``LikedSongs.predict()``, making it easy to visualize personalized
    recommendations.
 
    Parameters
    ----------
    scores_df : pd.DataFrame
        Output of ``LikedSongs.predict()``. Must contain 'track_name',
        'artists', and 'preference_score' columns.
    top_n : int
        Number of top-scoring tracks to display. Defaults to 20.
 
    Returns
    -------
    fig, ax
    """
    for col in ["track_genre", "tempo"]:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain a '{col}' column.")
 
    tmp = df[["track_genre", "tempo"]].copy()
    tmp["tempo"] = pd.to_numeric(tmp["tempo"], errors="coerce")
    tmp = tmp.dropna(subset=["tempo", "track_genre"])
 
    top_genres = tmp["track_genre"].value_counts().head(top_n).index.tolist()
    tmp = tmp[tmp["track_genre"].isin(top_genres)]
 
    # Use GroupBy to build ordered list of tempo arrays for boxplot
    grouped = tmp.groupby("track_genre")["tempo"].apply(list)
    means = tmp.groupby("track_genre")["tempo"].mean()
    ordered = means.reindex(top_genres).sort_values(ascending=False).index.tolist()
    data = [grouped[g] for g in ordered]
 
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, patch_artist=True, vert=True)
 
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(ordered)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
 
    ax.set_xticks(range(1, len(ordered) + 1))
    ax.set_xticklabels(ordered, rotation=45, ha="right")
    ax.set_ylabel("Tempo (BPM)")
    ax.set_xlabel("Genre")
    ax.set_title(f"Tempo Distribution by Genre (Top {top_n} Genres)")
 
    Example
    -------
    >>> from unwrapped.io import load_tracks
    >>> from unwrapped.preference import LikedSongs
    >>> from unwrapped.visualization import plot_preference_scores, save_figure
    >>>
    >>> df = load_tracks("data/raw/spotify_data.csv")
    >>> liked = LikedSongs(df)
    >>> liked.add_by_name("Blinding Lights")
    >>> scores = liked.predict(top_n=50)
    >>> fig, ax = plot_preference_scores(scores)
    >>> save_figure(fig, "outputs/my_recommendations.png")
    """
    required = ["track_name", "artists", "preference_score"]
    missing = [c for c in required if c not in scores_df.columns]
    if missing:
        raise ValueError(f"scores_df is missing columns: {missing}")
 
    top = (
        scores_df
        .nlargest(top_n, "preference_score")
        .sort_values("preference_score")
        .copy()
    )
 
    # Build readable labels: "Track Name — Artist"
    top["label"] = top["track_name"].str.strip() + "  —  " + top["artists"].str.strip()
 
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    bars = ax.barh(
        top["label"],
        top["preference_score"],
        color="#1DB954",
        alpha=0.85,
    )
 
    # Annotate score values at end of each bar
    for bar, score in zip(bars, top["preference_score"]):
        ax.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            fontsize=8,
        )
 
    ax.set_xlim(0, top["preference_score"].max() * 1.15)
    ax.set_xlabel("Preference Score")
    ax.set_title(f"Top {top_n} Recommended Tracks")
    fig.tight_layout()
 
    return fig, ax