"""Popularity regression module for predicting Spotify track popularity.

This module validates the input data, handles missing values, compares a
baseline linear regression model to a random forest regressor, reports
feature importance, and optionally saves outputs for downstream use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import argparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

from .io import load_data


DEFAULT_OUTPUT_DIR = "outputs"

REQUIRED_COLUMNS: list[str] = [
    "popularity",
    "track_genre",
    "duration_ms",
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


def validate_data(df: pd.DataFrame) -> None:
    """Check that ``df`` is non-empty and contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Spotify dataset to validate.

    Raises
    ------
    ValueError
        If ``df`` is empty or is missing any column in :data:`REQUIRED_COLUMNS`.
    """
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

# NOTE: Retained for testing and modular use.
# Not used in run_popularity_pipeline to avoid data leakage.
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows without a popularity target and impute the rest.

    Rows missing the ``popularity`` target are dropped. Missing
    ``track_genre`` values are replaced with ``"unknown"``. Numeric
    feature columns are filled with their median.

    Parameters
    ----------
    df : pd.DataFrame
        Validated dataset.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with missing values handled.
    """
    df = df.copy()

    df = df.dropna(subset=["popularity"])
    df["track_genre"] = df["track_genre"].fillna("unknown")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_feature_cols = [col for col in numeric_cols if col != "popularity"]

    for col in numeric_feature_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns and one-hot encode ``track_genre``.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset already passed through :func:`handle_missing_values`.

    Returns
    -------
    pd.DataFrame
        Model-ready feature frame.
    """
    df = df.copy()

    cols_to_drop = [
        "track_id",
        "track_name",
        "album_name",
        "artists",
        "Unnamed: 0",
    ]

    df = df.drop(columns=cols_to_drop, errors="ignore")
    df = pd.get_dummies(df, columns=["track_genre"], drop_first=True)

    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split ``df`` into train/test feature matrices and popularity targets.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset containing a ``popularity`` column.
    test_size : float, default 0.2
        Fraction of rows held out for testing.
    random_state : int, default 42
        Seed passed to ``train_test_split`` for reproducibility.

    Returns
    -------
    tuple
        ``(X_train, X_test, y_train, y_test)``.
    """
    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def train_linear_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Fit a baseline linear regression on the training set."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame, y_train: pd.Series
) -> RandomForestRegressor:
    """Fit a tuned random forest regressor on the training set."""
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict[str, Any]:
    """Score ``model`` on the held-out test set and print a short report.

    Parameters
    ----------
    model : Any
        Fitted scikit-learn estimator exposing ``.predict``.
    X_test, y_test :
        Held-out features and target.
    model_name : str
        Label used in the printed report.

    Returns
    -------
    dict
        ``{"model", "rmse", "mae", "r2"}`` for downstream comparison.
    """
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"{model_name} RMSE: {rmse:.2f}")
    print(f"{model_name} MAE: {mae:.2f}")
    print(f"{model_name} R2: {r2:.4f}")

    return {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def cross_validate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> dict[str, float]:
    """Run k-fold cross-validation and return RMSE and R^2 summaries.

    Parameters
    ----------
    model : Any
        Unfitted scikit-learn estimator.
    X_train, y_train :
        Training split used for cross-validation.
    cv : int, default 5
        Number of folds.

    Returns
    -------
    dict
        ``cv_rmse_mean``, ``cv_rmse_std``, and ``cv_r2_mean``.
    """
    neg_rmse_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )

    r2_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=1,
    )

    cv_rmse_mean = -neg_rmse_scores.mean()
    cv_rmse_std = neg_rmse_scores.std()
    cv_r2_mean = r2_scores.mean()

    return {
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "cv_r2_mean": cv_r2_mean,
    }


def compare_models(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Combine per-model result dicts into a sorted comparison DataFrame."""
    comparison_df = pd.DataFrame(results)
    numeric_cols = comparison_df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        comparison_df[col] = comparison_df[col].round(4)

    comparison_df = comparison_df.sort_values(by="rmse").reset_index(drop=True)

    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    return comparison_df


def get_feature_importance(
    model: RandomForestRegressor,
    X_train: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top ``top_n`` features ranked by ``feature_importances_``."""
    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    })

    importances = (
        importances.sort_values(by="importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print("\nTop Feature Importances:")
    print(importances.to_string(index=False))

    return importances


def save_outputs(
    comparison_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, str]:
    """Write model comparison, feature importance, and predictions to CSV.

    Parameters
    ----------
    comparison_df, feature_importance_df, predictions_df :
        Artifacts produced by :func:`run_popularity_pipeline`.
    output_dir : str | Path, default :data:`DEFAULT_OUTPUT_DIR`
        Directory to write into; created if missing.

    Returns
    -------
    dict[str, str]
        Mapping of artifact name to the path written.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_path = output_path / "popularity_model_comparison.csv"
    importance_path = output_path / "popularity_feature_importance.csv"
    predictions_path = output_path / "popularity_predictions.csv"

    comparison_df.to_csv(comparison_path, index=False)
    feature_importance_df.to_csv(importance_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    return {
        "comparison": str(comparison_path),
        "feature_importance": str(importance_path),
        "predictions": str(predictions_path),
    }


def run_popularity_pipeline(
    data_path: str = "data/raw/spotify_data.csv",
    save_results: bool = True,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Run the full popularity modeling pipeline end-to-end.

    Parameters
    ----------
    data_path : str
        CSV path passed to :func:`unwrapped.io.load_data`.
    save_results : bool, default True
        When ``True``, call :func:`save_outputs` to write CSVs.
    output_dir : str | Path
        Directory the CSV artifacts are written to when ``save_results``
        is ``True``.

    Returns
    -------
    dict
        Fitted models, comparison frame, feature importances, and the
        test-set predictions.
    """
    df = load_data(data_path)

    validate_data(df)

    # --- Step 1: Drop rows missing target ---
    df = df.dropna(subset=["popularity"]).copy()

    # --- Step 2: Split BEFORE imputation ---
    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Step 3: Handle categorical missing values ---
    X_train["track_genre"] = X_train["track_genre"].fillna("unknown")
    X_test["track_genre"] = X_test["track_genre"].fillna("unknown")

    # --- Step 4: Impute numeric columns using TRAINING medians ---
    numeric_cols = X_train.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        train_median = X_train[col].median()
        X_train[col] = X_train[col].fillna(train_median)
        X_test[col] = X_test[col].fillna(train_median)

    # --- Step 5: Recombine with target for preprocessing ---
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # --- Step 6: One-hot encode separately ---
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # --- Step 7: Split again into X/y ---
    X_train = train_df.drop(columns=["popularity"])
    y_train = train_df["popularity"]

    X_test = test_df.drop(columns=["popularity"])
    y_test = test_df["popularity"]

    # --- Step 8: Align columns (IMPORTANT for dummies) ---
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    linear_model = train_linear_model(X_train, y_train)
    linear_results = evaluate_model(
            linear_model,
            X_test,
            y_test,
            "Linear Regression",
        )

    linear_cv = cross_validate_model(LinearRegression(), X_train, y_train)
    linear_results.update(linear_cv)

    random_forest_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(
            random_forest_model,
            X_test,
            y_test,
            "Random Forest",
        )

    rf_cv = cross_validate_model(
            RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                max_features="sqrt",
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            ),
            X_train,
            y_train,
        )
    rf_results.update(rf_cv)

    comparison_df = compare_models([linear_results, rf_results])
    feature_importance_df = get_feature_importance(random_forest_model, X_train)

    test_predictions = pd.DataFrame({
            "actual_popularity": y_test.values,
            "linear_prediction": linear_model.predict(X_test),
            "random_forest_prediction": random_forest_model.predict(X_test),
        })

    if save_results:
            save_outputs(
                comparison_df,
                feature_importance_df,
                test_predictions,
                output_dir=output_dir,
            )

    return {
            "linear_model": linear_model,
            "random_forest_model": random_forest_model,
            "comparison": comparison_df,
            "feature_importance": feature_importance_df,
            "predictions": test_predictions,
        }


import sys

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the Spotify popularity prediction pipeline."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        default="data/raw/spotify_data.csv",
        help="Path to the Spotify dataset CSV.",
    )

    args = parser.parse_args(argv if argv is not None else [])

    try:
        run_popularity_pipeline(args.data_path)
    except FileNotFoundError:
        print(
            f"Error: data file not found at '{args.data_path}'.\n"
            "Provide a valid dataset path as a command-line argument or place "
            "the dataset in data/raw/spotify_data.csv."
        )
        raise SystemExit(1)
if __name__ == "__main__":
    main(sys.argv[1:])