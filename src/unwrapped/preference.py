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

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity


# The 9 continuous audio features used to build the taste profile.
# These are kept as a module-level constant so other modules can import
# and reuse the same feature set.
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

    def add_by_name(self, track_name: str, artist: str = None) -> None:
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

        Raises
        ------
        ValueError
            If no matching track is found.
        """
        mask = self.df["track_name"].str.lower() == track_name.lower()
        if artist:
            mask &= self.df["artists"].str.lower().str.contains(artist.lower())

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

    def predict(self, top_n: int = None, exclude_liked: bool = True) -> pd.DataFrame:
        """
        Score every song in the dataset by similarity to the taste profile.

        Uses cosine similarity between each song's feature vector and the
        mean feature vector of liked songs. Scores are min-max normalized
        to a 0–1 range.

        Parameters
        ----------
        top_n : int, optional
            If provided, return only the top N highest-scoring songs.
        exclude_liked : bool, default True
            If True, liked songs are excluded from the results (they'd
            trivially score near the top).

        Returns
        -------
        pd.DataFrame
            Dataset with an added 'preference_score' column (0–1),
            sorted descending. Columns: track_id, track_name, artists,
            track_genre, popularity, preference_score.
        """
        profile = self.build_profile()

        # Normalize all feature vectors and the profile onto the same scale.
        # loudness is negative and tempo is in the hundreds — raw cosine
        # similarity without scaling would be dominated by those columns.
        scaler = MinMaxScaler()
        feature_matrix = self.df[AUDIO_FEATURES].copy()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)

        profile_scaled = scaler.transform(profile.values.reshape(1, -1))

        # Cosine similarity: shape is (n_songs, 1)
        similarities = cosine_similarity(feature_matrix_scaled, profile_scaled).flatten()

        result = self.df.copy()
        result["preference_score"] = similarities

        # Normalize scores to 0–1
        min_score = result["preference_score"].min()
        max_score = result["preference_score"].max()
        if max_score > min_score:
            result["preference_score"] = (
                (result["preference_score"] - min_score) / (max_score - min_score)
            )
        else:
            result["preference_score"] = 0.0

        result["preference_score"] = result["preference_score"].round(4)

        if exclude_liked:
            result = result[~result["track_id"].isin(self.liked_ids)]

        result = result.sort_values("preference_score", ascending=False).reset_index(drop=True)

        output_cols = ["track_id", "track_name", "artists", "track_genre", "popularity", "preference_score"]
        result = result[output_cols]

        if top_n is not None:
            return result.head(top_n)

        return result

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
        """
        with open(filepath, "w") as f:
            json.dump(list(self.liked_ids), f, indent=2)

    def load(self, filepath: str) -> None:
        """
        Load liked song IDs from a previously saved JSON file.

        Validates that each loaded ID exists in the current dataset.
        IDs not found in the dataset are skipped with a warning.

        Parameters
        ----------
        filepath : str
            Path to the JSON file to load.
        """
        with open(filepath, "r") as f:
            ids = json.load(f)

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
