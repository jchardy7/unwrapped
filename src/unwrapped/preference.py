"""
preference.py
-------------
Tools for assembling a liked songs list and predicting preference scores
for tracks based on audio feature similarity.

Workflow:
    1. Create a LikedSongs object
    2. Add songs by track_id or track_name
    3. Call predict() to score all songs in your dataset

Example:
    from unwrapped.io import load_data
    from unwrapped.preference import LikedSongs

    df = load_data("spotify_data.csv")

    liked = LikedSongs(df)
    liked.add_by_id("5SuOikwiRyPMVoIQDJUgSV")
    liked.add_by_name("Blinding Lights")

    scores = liked.predict()
    print(scores.head(10))
"""

import json
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from .constants import AUDIO_FEATURES

ScoringMethod = Literal["cosine", "euclidean"]
WeightingScheme = Literal["uniform", "inverse_variance"]

_VARIANCE_EPS = 1e-9


class LikedSongs:
    """
    Manages a list of liked songs and predicts preference scores for a dataset.

    The preference score is the cosine similarity between each song's audio
    feature vector and the mean feature vector of all liked songs. Scores
    are normalized to a 0–1 range.

    Parameters
    ----------
    df : pd.DataFrame
        Full Spotify tracks DataFrame, as returned by unwrapped.io.load_data().
        Must contain the columns in AUDIO_FEATURES plus 'track_id' and 'track_name'.

    Attributes
    ----------
    liked_ids : set
        Set of track_id strings currently in the liked songs list.
    """

    def __init__(self, df: pd.DataFrame):
        self._validate_dataframe(df)
        # Drop duplicate track_ids so the feature matrix is clean
        self.df = df.drop_duplicates(subset="track_id").reset_index(drop=True)
        self.liked_ids: set = set()

    # ------------------------------------------------------------------
    # Adding and removing songs
    # ------------------------------------------------------------------

    def add_by_id(self, track_id: str) -> None:
        """
        Add a song to the liked list by its Spotify track ID.

        Parameters
        ----------
        track_id : str
            The Spotify track ID (e.g., '5SuOikwiRyPMVoIQDJUgSV').

        Raises
        ------
        ValueError
            If the track_id is not found in the dataset.
        """
        if track_id not in self.df["track_id"].values:
            raise ValueError(
                f"track_id '{track_id}' not found in dataset. "
                "Check the ID or make sure you loaded the right file."
            )
        self.liked_ids.add(track_id)

    def add_by_name(self, track_name: str, artist: str | None = None) -> None:
        """
        Add a song to the liked list by track name (case-insensitive).

        If multiple tracks share the same name, the first match is used.
        Pass `artist` to disambiguate.

        Parameters
        ----------
        track_name : str
            The name of the track to search for.
        artist : str, optional
            Artist name to narrow the search when there are duplicates.
            Matched as a case-insensitive substring (not a regex), so
            names containing ``(``, ``.``, etc. are safe.

        Raises
        ------
        ValueError
            If no matching track is found.
        """
        names = self.df["track_name"].fillna("").str.lower()
        mask = names == track_name.lower()
        if artist:
            # regex=False keeps special characters in artist names (parens,
            # dots, plus signs) from being interpreted as regex metachars.
            mask &= (
                self.df["artists"]
                .fillna("")
                .str.lower()
                .str.contains(artist.lower(), regex=False, na=False)
            )

        matches = self.df[mask]
        if matches.empty:
            artist_hint = f" by '{artist}'" if artist else ""
            raise ValueError(
                f"No track named '{track_name}'{artist_hint} found in dataset."
            )

        track_id = matches.iloc[0]["track_id"]
        self.liked_ids.add(track_id)

    def remove(self, track_id: str) -> None:
        """
        Remove a song from the liked list by track ID.

        Parameters
        ----------
        track_id : str
            The Spotify track ID to remove.
        """
        self.liked_ids.discard(track_id)

    def clear(self) -> None:
        """Remove all songs from the liked list."""
        self.liked_ids.clear()

    # ------------------------------------------------------------------
    # Inspecting the liked list
    # ------------------------------------------------------------------

    def show(self) -> pd.DataFrame:
        """
        Return a DataFrame of the currently liked songs.

        Returns
        -------
        pd.DataFrame
            Subset of the dataset containing only liked tracks, with
            columns: track_id, track_name, artists, popularity.
        """
        if not self.liked_ids:
            return pd.DataFrame(columns=["track_id", "track_name", "artists", "popularity"])

        liked_df = self.df[self.df["track_id"].isin(self.liked_ids)]
        return liked_df[["track_id", "track_name", "artists", "popularity"]].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.liked_ids)

    def __repr__(self) -> str:
        return f"LikedSongs({len(self.liked_ids)} songs)"

    # ------------------------------------------------------------------
    # Building the taste profile
    # ------------------------------------------------------------------

    def build_profile(self) -> pd.Series:
        """
        Compute the mean audio feature vector across all liked songs.

        This is the "taste profile" — the centroid in feature space that
        represents what the user tends to enjoy.

        Returns
        -------
        pd.Series
            Mean values for each feature in AUDIO_FEATURES.

        Raises
        ------
        ValueError
            If the liked list is empty.
        """
        if not self.liked_ids:
            raise ValueError(
                "Liked songs list is empty. Add songs with add_by_id() or add_by_name() first."
            )

        liked_df = self.df[self.df["track_id"].isin(self.liked_ids)]
        return liked_df[AUDIO_FEATURES].mean()

    # ------------------------------------------------------------------
    # Predicting preference scores
    # ------------------------------------------------------------------

    def predict(
        self,
        top_n: int | None = None,
        exclude_liked: bool = True,
        method: ScoringMethod = "cosine",
        weights: WeightingScheme = "uniform",
        return_explanations: bool = False,
        deduplicate: bool = True,
    ) -> pd.DataFrame:
        """
        Score every song in the dataset by similarity to the taste profile.

        Two scoring methods are supported:

        * ``"cosine"`` (default, backward compatible): cosine similarity
          between each song's min-max scaled feature vector and the
          scaled taste profile.  Raw similarities are then min-max
          normalized to 0–1.
        * ``"euclidean"``: ``-Σ w_i · (track_i − profile_i)²`` in the
          scaled feature space, where ``w_i`` comes from ``weights``.
          Raw scores are min-max normalized to 0–1 for the output.
          The raw, un-normalized score decomposes exactly into per-feature
          attributions returned by :meth:`explain`.

        Parameters
        ----------
        top_n : int, optional
            If provided, return only the top ``top_n`` highest-scoring songs.
        exclude_liked : bool, default True
            If True, liked songs are excluded from the results (they'd
            trivially score near the top).
        method : {"cosine", "euclidean"}, default "cosine"
            Scoring backend.  Cosine preserves the original behavior;
            euclidean gives an exact per-feature decomposition.
        weights : {"uniform", "inverse_variance"}, default "uniform"
            Only used for ``method="euclidean"``.  ``"inverse_variance"``
            down-weights features where the liked set is inconsistent and
            up-weights features the user is consistent on.  Falls back to
            uniform (with a warning) when fewer than two songs are liked.
        return_explanations : bool, default False
            When True, add a ``top_matches`` column summarizing the three
            features that contributed most to each track's score.
        deduplicate : bool, default True
            When True, collapse rows that share the same ``(track_name,
            artists)`` pair, keeping only the highest-scoring row.  The
            Spotify dataset stores the same song once per genre tag, so
            a single song can otherwise dominate the top N.

        Returns
        -------
        pd.DataFrame
            Dataset sorted by ``preference_score`` descending.  Columns:
            ``track_id, track_name, artists, track_genre, popularity,
            preference_score`` (plus ``top_matches`` when requested).
        """
        scaled_matrix, profile_scaled = self._scaled_feature_space()
        raw_scores = self._raw_scores(
            scaled_matrix=scaled_matrix,
            profile_scaled=profile_scaled,
            method=method,
            weights=weights,
        )

        normalized = self._minmax(raw_scores)

        result = self.df.copy()
        result["preference_score"] = np.round(normalized, 4)

        if return_explanations:
            weight_vector = self._compute_weights(weights)
            deltas = scaled_matrix - profile_scaled
            result["top_matches"] = self._describe_top_matches(
                deltas=deltas, weights=weight_vector, method=method
            )

        if exclude_liked:
            result = result[~result["track_id"].isin(self.liked_ids)]

        result = result.sort_values(
            "preference_score", ascending=False
        ).reset_index(drop=True)

        if deduplicate:
            # Keep the highest-scoring row for each (track_name, artists)
            # pair. drop_duplicates keeps the first occurrence, and the
            # frame is already sorted by score descending.
            result = (
                result.dropna(subset=["track_name", "artists"])
                .drop_duplicates(subset=["track_name", "artists"])
                .reset_index(drop=True)
            )

        output_cols = [
            "track_id",
            "track_name",
            "artists",
            "track_genre",
            "popularity",
            "preference_score",
        ]
        if return_explanations:
            output_cols.append("top_matches")
        result = result[output_cols]

        if top_n is not None:
            return result.head(top_n)

        return result

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------

    def explain(
        self,
        track_id: str,
        method: ScoringMethod = "cosine",
        weights: WeightingScheme = "uniform",
    ) -> pd.DataFrame:
        """
        Break a track's score down into per-feature contributions.

        Returns one row per audio feature with the raw and scaled values
        for both the track and the taste profile, the signed and absolute
        delta between them, and the weight used for that feature.  When
        ``method="euclidean"``, an additional ``attribution`` column is
        included; summing it reproduces the track's raw (un-normalized)
        score.

        Rows are sorted by ``abs_delta`` ascending so the strongest
        matches appear first.

        Parameters
        ----------
        track_id : str
            Spotify track ID to explain.  Must exist in the dataset.
        method : {"cosine", "euclidean"}
            Must match the method used to compute the score being
            explained.  Only affects which columns are returned.
        weights : {"uniform", "inverse_variance"}
            Weighting scheme applied when ``method="euclidean"``.

        Raises
        ------
        ValueError
            If ``track_id`` is not in the dataset, or the liked list is
            empty so no profile can be built.
        """
        if not self.liked_ids:
            raise ValueError(
                "Liked songs list is empty. Add songs with add_by_id() or "
                "add_by_name() before asking for an explanation."
            )
        track_rows = self.df.index[self.df["track_id"] == track_id]
        if len(track_rows) == 0:
            raise ValueError(f"track_id '{track_id}' not found in dataset.")
        row_idx = int(track_rows[0])

        scaled_matrix, profile_scaled = self._scaled_feature_space()
        track_scaled = scaled_matrix[row_idx]

        profile_raw = self.build_profile().reindex(AUDIO_FEATURES)
        track_raw = self.df.loc[row_idx, AUDIO_FEATURES].astype(float)

        weight_vector = self._compute_weights(weights)
        delta = track_scaled - profile_scaled.flatten()

        explanation = pd.DataFrame({
            "feature": AUDIO_FEATURES,
            "track_raw": track_raw.values,
            "profile_raw": profile_raw.values,
            "track_scaled": track_scaled,
            "profile_scaled": profile_scaled.flatten(),
            "delta": delta,
            "abs_delta": np.abs(delta),
            "weight": weight_vector,
        })

        if method == "euclidean":
            # Per-feature contribution to the raw (negative squared
            # distance) score.  sum(attribution) == raw score exactly.
            explanation["attribution"] = -weight_vector * delta ** 2

        return explanation.sort_values("abs_delta").reset_index(drop=True)

    def explain_top(
        self,
        track_id: str,
        n: int = 3,
        method: ScoringMethod = "cosine",
        weights: WeightingScheme = "uniform",
    ) -> dict[str, pd.DataFrame]:
        """
        Return the ``n`` features that match best and worst for a track.

        A thin wrapper around :meth:`explain` that slices the head and
        tail of the sorted explanation.

        Parameters
        ----------
        track_id : str
            Spotify track ID to summarize.
        n : int, default 3
            Number of features to include in each bucket.
        method : {"cosine", "euclidean"}
            Passed through to :meth:`explain`.
        weights : {"uniform", "inverse_variance"}
            Passed through to :meth:`explain`.

        Returns
        -------
        dict[str, pd.DataFrame]
            ``{"matches": ..., "mismatches": ...}`` — the ``n`` features
            with the smallest and largest ``abs_delta`` respectively.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        explanation = self.explain(track_id, method=method, weights=weights)
        return {
            "matches": explanation.head(n).reset_index(drop=True),
            "mismatches": (
                explanation.tail(n)
                .sort_values("abs_delta", ascending=False)
                .reset_index(drop=True)
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _scaled_feature_space(self) -> tuple[np.ndarray, np.ndarray]:
        """Min-max scale the feature matrix and the taste profile together.

        Returns the scaled feature matrix (shape ``n_tracks × n_features``)
        and the scaled profile vector (shape ``1 × n_features``).
        """
        profile = self.build_profile()

        scaler = MinMaxScaler()
        feature_matrix = self.df[AUDIO_FEATURES].copy()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        # DataFrame wrapper keeps sklearn from warning about lost feature names.
        profile_df = profile.reindex(AUDIO_FEATURES).to_frame().T
        profile_scaled = scaler.transform(profile_df)

        return feature_matrix_scaled, profile_scaled

    def _raw_scores(
        self,
        scaled_matrix: np.ndarray,
        profile_scaled: np.ndarray,
        method: ScoringMethod,
        weights: WeightingScheme,
    ) -> np.ndarray:
        """Dispatch to the requested scoring method and return raw scores."""
        if method == "cosine":
            return cosine_similarity(scaled_matrix, profile_scaled).flatten()
        if method == "euclidean":
            weight_vector = self._compute_weights(weights)
            deltas = scaled_matrix - profile_scaled
            return -np.sum(weight_vector * deltas ** 2, axis=1)
        raise ValueError(
            f"Unknown method '{method}'. Expected 'cosine' or 'euclidean'."
        )

    def _compute_weights(self, weights: WeightingScheme) -> np.ndarray:
        """Return the per-feature weights normalized to sum to 1.

        ``"uniform"`` gives every feature equal weight.
        ``"inverse_variance"`` uses ``1 / (var + eps)`` of the liked
        songs' scaled features, so features the user is consistent on
        dominate the score.  Falls back to uniform (with a warning)
        when fewer than two songs are liked.
        """
        n_features = len(AUDIO_FEATURES)
        if weights == "uniform":
            return np.full(n_features, 1.0 / n_features)
        if weights == "inverse_variance":
            if len(self.liked_ids) < 2:
                warnings.warn(
                    "inverse_variance weighting needs at least 2 liked songs; "
                    "falling back to uniform weights.",
                    stacklevel=3,
                )
                return np.full(n_features, 1.0 / n_features)

            liked_df = self.df[self.df["track_id"].isin(self.liked_ids)]
            # Fit the scaler on the full dataset so the variance is
            # measured in the same space the predictions live in.
            scaler = MinMaxScaler()
            scaler.fit(self.df[AUDIO_FEATURES])
            liked_scaled = scaler.transform(liked_df[AUDIO_FEATURES])

            variances = liked_scaled.var(axis=0, ddof=0)
            inv = 1.0 / (variances + _VARIANCE_EPS)
            return inv / inv.sum()
        raise ValueError(
            f"Unknown weights '{weights}'. Expected 'uniform' or 'inverse_variance'."
        )

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        """Min-max normalize a 1-D score array to the 0–1 range."""
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi > lo:
            return (values - lo) / (hi - lo)
        return np.zeros_like(values, dtype=float)

    @staticmethod
    def _describe_top_matches(
        deltas: np.ndarray,
        weights: np.ndarray,
        method: ScoringMethod,
        n: int = 3,
    ) -> list[str]:
        """One-line ``top_matches`` string per row for :meth:`predict`.

        For cosine we rank by raw absolute delta (smaller = closer); for
        euclidean we rank by per-feature attribution (less negative = closer).
        Returns the ``n`` best-matching feature names per row, joined with
        commas.
        """
        abs_deltas = np.abs(deltas)
        if method == "euclidean":
            abs_deltas = weights * abs_deltas ** 2  # non-negative ranking key

        # argsort gives indices of the smallest values first.
        top_idx = np.argsort(abs_deltas, axis=1)[:, :n]
        feature_array = np.array(AUDIO_FEATURES)
        return [", ".join(feature_array[row].tolist()) for row in top_idx]

    # ------------------------------------------------------------------
    # Saving and loading the liked list
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """
        Save the liked song IDs to a JSON file.

        Parameters
        ----------
        filepath : str
            Path to write the JSON file (e.g., 'my_likes.json').

        Raises
        ------
        OSError
            If the file cannot be written (permissions, disk full, etc.).
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(sorted(self.liked_ids), f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load liked song IDs from a previously saved JSON file.

        Validates that each loaded ID exists in the current dataset.
        IDs not found in the dataset are skipped with a warning.

        Parameters
        ----------
        filepath : str
            Path to the JSON file to load.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file does not contain a JSON list of track IDs.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Liked songs file not found: {filepath}")

        with path.open("r") as f:
            try:
                ids = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Could not parse liked songs file '{filepath}': {exc}"
                ) from exc

        if not isinstance(ids, list):
            raise ValueError(
                f"Liked songs file '{filepath}' must contain a JSON list of track IDs."
            )

        skipped = 0
        for track_id in ids:
            if track_id in self.df["track_id"].values:
                self.liked_ids.add(track_id)
            else:
                skipped += 1

        if skipped:
            print(f"Warning: {skipped} track ID(s) from file not found in dataset and were skipped.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_dataframe(df: pd.DataFrame) -> None:
        """
        Check that the DataFrame has all required columns.

        Raises
        ------
        ValueError
            If any required column is missing.
        """
        required = set(AUDIO_FEATURES) | {"track_id", "track_name", "artists"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing)}. "
                "Make sure you loaded the data using unwrapped.io.load_data()."
            )
