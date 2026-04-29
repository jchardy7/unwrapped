"""Multi-class genre classifier for Spotify tracks.

Predicts ``track_genre`` from audio features. Mirrors the structure of
:mod:`unwrapped.hit_shape_predictor` so the project stays internally
consistent: validate, prepare, split, train, evaluate, compare, save.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .constants import AUDIO_FEATURES_FOR_HIT_CLASSIFICATION
from .io import load_data


REQUIRED_COLUMNS: list[str] = AUDIO_FEATURES_FOR_HIT_CLASSIFICATION + ["track_genre"]

RF_PARAMS: dict = {
    "n_estimators": 300,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": 1,
    "class_weight": "balanced",
}

LR_PARAMS: dict = {
    "max_iter": 1000,
    "random_state": 42,
    "class_weight": "balanced",
    "solver": "lbfgs",
}


def validate_data(df: pd.DataFrame) -> None:
    """Validate that ``df`` is non-empty and has the audio features + genre."""
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def prepare_genre_data(
    df: pd.DataFrame, min_samples_per_genre: int = 50
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rare genres, coerce dtypes, and return feature matrix + target.

    Parameters
    ----------
    df : pd.DataFrame
        Validated dataset.
    min_samples_per_genre : int, default 50
        Genres with fewer rows than this are removed (they hurt stratified
        splits and macro-F1).

    Returns
    -------
    tuple
        ``(X, y)`` ready for ``train_test_split``.
    """
    df = df.copy()
    df = df.dropna(subset=["track_genre"])

    counts = df["track_genre"].value_counts()
    keep = counts[counts >= min_samples_per_genre].index
    df = df[df["track_genre"].isin(keep)].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            f"No genres have at least {min_samples_per_genre} samples."
        )

    if "explicit" in df.columns:
        df["explicit"] = df["explicit"].fillna(False).astype(int)

    feature_cols = [c for c in AUDIO_FEATURES_FOR_HIT_CLASSIFICATION if c in df.columns]
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols].astype(float)
    y = df["track_genre"].astype(str)
    return X, y


def split_genre_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the genre target."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_logistic_genre_classifier(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Pipeline:
    """Fit a multinomial logistic regression with feature scaling."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(**LR_PARAMS)),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest_genre_classifier(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestClassifier:
    """Fit a multi-class random forest classifier."""
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    return model


def top_k_accuracy(model: Any, X: pd.DataFrame, y: pd.Series, k: int = 3) -> float:
    """Fraction of samples where the true label is among the top-``k`` probas.

    Falls back to plain ``accuracy_score`` when the model exposes fewer than
    ``k`` classes.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError("Model must implement predict_proba for top_k_accuracy.")

    classes = np.asarray(model.classes_)
    k = min(k, len(classes))

    probabilities = model.predict_proba(X)
    top_k_idx = np.argsort(probabilities, axis=1)[:, -k:]
    top_k_labels = classes[top_k_idx]

    y_arr = np.asarray(y)[:, None]
    matches = (top_k_labels == y_arr).any(axis=1)
    return float(matches.mean())


def evaluate_genre_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    top_k: int = 3,
) -> dict[str, Any]:
    """Score ``model`` on the held-out set with multi-class metrics."""
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro = f1_score(y_test, predictions, average="macro", zero_division=0)
    weighted = f1_score(y_test, predictions, average="weighted", zero_division=0)
    top_k_score = top_k_accuracy(model, X_test, y_test, k=top_k)

    print(f"{model_name} accuracy: {accuracy:.4f}")
    print(f"{model_name} macro F1: {macro:.4f}")
    print(f"{model_name} weighted F1: {weighted:.4f}")
    print(f"{model_name} top-{top_k} accuracy: {top_k_score:.4f}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "top_k_accuracy": top_k_score,
        "macro_f1": macro,
        "weighted_f1": weighted,
    }


def compare_genre_models(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a tidy comparison frame sorted by macro F1 (best first)."""
    comparison = pd.DataFrame(results)
    numeric = comparison.select_dtypes(include=["number"]).columns
    for col in numeric:
        comparison[col] = comparison[col].round(4)
    comparison = comparison.sort_values(by="macro_f1", ascending=False).reset_index(
        drop=True
    )

    print("\nGenre Model Comparison:")
    print(comparison.to_string(index=False))
    return comparison


def confusion_matrix_df(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series
) -> pd.DataFrame:
    """Return the confusion matrix as a DataFrame indexed by class label."""
    predictions = model.predict(X_test)
    labels = sorted(set(np.unique(y_test)) | set(np.unique(predictions)))
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def save_outputs(
    comparison_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_dir: str | Path = "outputs",
) -> dict[str, str]:
    """Write the three genre-classifier artifacts to ``output_dir``."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_path = output_path / "genre_model_comparison.csv"
    confusion_path = output_path / "genre_confusion_matrix.csv"
    predictions_path = output_path / "genre_predictions.csv"

    comparison_df.to_csv(comparison_path, index=False)
    confusion_df.to_csv(confusion_path)
    predictions_df.to_csv(predictions_path, index=False)

    return {
        "comparison": str(comparison_path),
        "confusion_matrix": str(confusion_path),
        "predictions": str(predictions_path),
    }


def run_genre_classifier_pipeline(
    data_path: str = "data/raw/spotify_data.csv",
    min_samples_per_genre: int = 50,
    save_results: bool = True,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    """End-to-end pipeline: load, prep, split, train, evaluate, compare, save."""
    df = load_data(data_path)
    validate_data(df)

    X, y = prepare_genre_data(df, min_samples_per_genre=min_samples_per_genre)
    X_train, X_test, y_train, y_test = split_genre_data(X, y)

    logistic_model = train_logistic_genre_classifier(X_train, y_train)
    logistic_results = evaluate_genre_model(
        logistic_model, X_test, y_test, "Logistic Regression"
    )

    rf_model = train_random_forest_genre_classifier(X_train, y_train)
    rf_results = evaluate_genre_model(rf_model, X_test, y_test, "Random Forest")

    comparison_df = compare_genre_models([logistic_results, rf_results])
    confusion_df = confusion_matrix_df(rf_model, X_test, y_test)
    predictions_df = pd.DataFrame(
        {
            "actual_genre": y_test.values,
            "logistic_prediction": logistic_model.predict(X_test),
            "random_forest_prediction": rf_model.predict(X_test),
        }
    )

    if save_results:
        save_outputs(comparison_df, confusion_df, predictions_df, output_dir=output_dir)

    return {
        "logistic_model": logistic_model,
        "random_forest_model": rf_model,
        "comparison": comparison_df,
        "confusion_matrix": confusion_df,
        "predictions": predictions_df,
    }


def main() -> None:
    """Run the default genre-classifier pipeline."""
    run_genre_classifier_pipeline("data/raw/spotify_data.csv")


if __name__ == "__main__":
    main()
