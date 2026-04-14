"""
Hit shape predictor module for identifying whether a Spotify track
resembles the profile of a hit song.

This module defines a hit using a popularity threshold, builds average
audio-feature profiles for hit and non-hit songs, computes similarity-
based features, trains classification models, and reports performance.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, cross_val_score

from .io import load_data


REQUIRED_COLUMNS = [
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
]


AUDIO_FEATURE_COLUMNS = [
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
]


def validate_data(df):
    """Validate that the dataframe is not empty and contains needed columns."""
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )


def handle_missing_values(df):
    """Handle missing values in popularity and predictor columns."""
    df = df.copy()

    df = df.dropna(subset=["popularity"])

    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].fillna(False)

    numeric_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    numeric_feature_cols = [col for col in numeric_cols if col != "popularity"]

    for col in numeric_feature_cols:
        if col == "explicit":
            continue
        df[col] = df[col].fillna(df[col].median())

    return df


def preprocess_data(df):
    """Drop non-modeling identifier/text columns and standardize booleans."""
    df = df.copy()

    cols_to_drop = [
        "Unnamed: 0",
        "track_id",
        "artists",
        "album_name",
        "track_name",
        "track_genre",
    ]

    df = df.drop(columns=cols_to_drop, errors="ignore")

    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].astype(int)

    return df


def create_hit_label(df, threshold=70):
    """Create a binary hit column using a popularity threshold."""
    df = df.copy()
    df["is_hit"] = (df["popularity"] >= threshold).astype(int)
    return df


def get_audio_feature_columns():
    """Return the list of audio feature columns used in the model."""
    return AUDIO_FEATURE_COLUMNS.copy()


def build_hit_profiles(df):
    """
    Build average audio-feature profiles for hit and non-hit songs.

    Returns a dataframe indexed by hit label:
    - 0 = non-hit profile
    - 1 = hit profile
    """
    feature_cols = get_audio_feature_columns()

    profiles = (
        df.groupby("is_hit")[feature_cols]
        .mean()
        .sort_index()
    )

    if profiles.shape[0] < 2:
        raise ValueError("Both hit and non-hit groups are required to build profiles.")

    return profiles


def calculate_profile_differences(profiles):
    """Calculate hit minus non-hit feature differences."""
    if 0 not in profiles.index or 1 not in profiles.index:
        raise ValueError("Profiles must contain both non-hit (0) and hit (1).")

    difference_df = pd.DataFrame({
        "feature": profiles.columns,
        "non_hit_mean": profiles.loc[0].values,
        "hit_mean": profiles.loc[1].values,
        "difference_hit_minus_non_hit": (
            profiles.loc[1].values - profiles.loc[0].values
        )
    })

    difference_df["absolute_difference"] = (
        difference_df["difference_hit_minus_non_hit"].abs()
    )

    difference_df = difference_df.sort_values(
        by="absolute_difference",
        ascending=False
    ).reset_index(drop=True)

    return difference_df


def compute_centroids(df):
    """Return the non-hit centroid and hit centroid as numpy arrays."""
    profiles = build_hit_profiles(df)
    non_hit_centroid = profiles.loc[0].to_numpy()
    hit_centroid = profiles.loc[1].to_numpy()
    return non_hit_centroid, hit_centroid


def compute_similarity_features(df):
    """
    Compute engineered similarity features based on distance to
    hit and non-hit audio profiles.
    """
    df = df.copy()
    feature_cols = get_audio_feature_columns()

    non_hit_centroid, hit_centroid = compute_centroids(df)
    X = df[feature_cols].to_numpy()

    distance_to_non_hit = np.linalg.norm(X - non_hit_centroid, axis=1)
    distance_to_hit = np.linalg.norm(X - hit_centroid, axis=1)

    df["distance_to_non_hit"] = distance_to_non_hit
    df["distance_to_hit"] = distance_to_hit
    df["hit_distance_advantage"] = distance_to_non_hit - distance_to_hit
    df["hit_closeness_ratio"] = distance_to_hit / (distance_to_non_hit + 1e-9)

    return df


def build_modeling_dataframe(df):
    """Build the final modeling dataframe using engineered similarity features."""
    df = df.copy()

    model_df = df[
        [
            "distance_to_non_hit",
            "distance_to_hit",
            "hit_distance_advantage",
            "hit_closeness_ratio",
            "is_hit",
        ]
    ].copy()

    return model_df


def split_data(df, test_size=0.2, random_state=42):
    """Split similarity-feature data into train and test sets."""
    X = df.drop(columns=["is_hit"])
    y = df["is_hit"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def train_logistic_model(X_train, y_train):
    """Train a logistic regression classifier."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train a random forest classifier."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate classifier performance using several classification metrics."""
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    print(f"{model_name} F1: {f1:.4f}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def cross_validate_model(model, X_train, y_train, cv=5):
    """Cross-validate a model using accuracy and F1."""
    accuracy_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    f1_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    return {
        "cv_accuracy_mean": accuracy_scores.mean(),
        "cv_accuracy_std": accuracy_scores.std(),
        "cv_f1_mean": f1_scores.mean(),
    }


def compare_models(results):
    """Create a comparison dataframe for trained models."""
    comparison_df = pd.DataFrame(results)
    numeric_cols = comparison_df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        comparison_df[col] = comparison_df[col].round(4)

    comparison_df = comparison_df.sort_values(
        by="f1",
        ascending=False
    ).reset_index(drop=True)

    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    return comparison_df


def get_feature_importance(model, X_train):
    """Return feature importances from the random forest classifier."""
    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

    print("\nSimilarity Feature Importances:")
    print(importance_df.to_string(index=False))

    return importance_df


def build_predictions_df(model, X_test, y_test):
    """Build a dataframe of actual vs predicted hit labels."""
    predictions = model.predict(X_test)

    predictions_df = pd.DataFrame({
        "actual_is_hit": y_test.values,
        "predicted_is_hit": predictions,
    })

    return predictions_df


def save_outputs(
    profiles_df,
    differences_df,
    comparison_df,
    importance_df,
    predictions_df,
    output_dir="outputs"
):
    """Save output tables as CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    profiles_df.to_csv(output_path / "hit_shape_profiles.csv")
    differences_df.to_csv(output_path / "hit_shape_profile_differences.csv", index=False)
    comparison_df.to_csv(output_path / "hit_shape_model_comparison.csv", index=False)
    importance_df.to_csv(output_path / "hit_shape_similarity_importance.csv", index=False)
    predictions_df.to_csv(output_path / "hit_shape_predictions.csv", index=False)


def run_hit_shape_pipeline(
    data_path="data/raw/spotify_data.csv",
    hit_threshold=70,
    save_results=True
):
    """Run the full hit shape prediction pipeline."""
    df = load_data(data_path)

    validate_data(df)
    df = handle_missing_values(df)
    df = preprocess_data(df)
    df = create_hit_label(df, threshold=hit_threshold)

    profiles_df = build_hit_profiles(df)
    differences_df = calculate_profile_differences(profiles_df)

    similarity_df = compute_similarity_features(df)
    model_df = build_modeling_dataframe(similarity_df)

    X_train, X_test, y_train, y_test = split_data(model_df)

    logistic_model = train_logistic_model(X_train, y_train)
    logistic_results = evaluate_model(
        logistic_model,
        X_test,
        y_test,
        "Logistic Regression"
    )
    logistic_results.update(
        cross_validate_model(
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            ),
            X_train,
            y_train
        )
    )

    random_forest_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest"
    )
    rf_results.update(
        cross_validate_model(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                max_features="sqrt",
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            ),
            X_train,
            y_train
        )
    )

    comparison_df = compare_models([logistic_results, rf_results])
    importance_df = get_feature_importance(random_forest_model, X_train)
    predictions_df = build_predictions_df(random_forest_model, X_test, y_test)

    if save_results:
        save_outputs(
            profiles_df,
            differences_df,
            comparison_df,
            importance_df,
            predictions_df
        )

    return {
        "profiles": profiles_df,
        "differences": differences_df,
        "comparison": comparison_df,
        "feature_importance": importance_df,
        "predictions": predictions_df,
        "logistic_model": logistic_model,
        "random_forest_model": random_forest_model,
    }


def main():
    """Run the default hit shape pipeline."""
    run_hit_shape_pipeline("data/raw/spotify_data.csv")


if __name__ == "__main__":
    main()