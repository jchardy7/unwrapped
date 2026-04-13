"""
Popularity regression module for predicting Spotify track popularity
using audio features and metadata.

This module validates the input data, handles missing values, compares
a baseline linear regression model to a random forest regressor, reports
feature importance, and optionally saves outputs for downstream use.
"""

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from .io import load_data


REQUIRED_COLUMNS = [
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
    "tempo"
]


def validate_data(df):
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )


def handle_missing_values(df):
    df = df.copy()

    df = df.dropna(subset=["popularity"])
    df["track_genre"] = df["track_genre"].fillna("unknown")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_feature_cols = [col for col in numeric_cols if col != "popularity"]

    for col in numeric_feature_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def preprocess_data(df):
    df = df.copy()

    cols_to_drop = [
        "track_id",
        "track_name",
        "album_name",
        "artists",
        "Unnamed: 0"
    ]

    df = df.drop(columns=cols_to_drop, errors="ignore")
    df = pd.get_dummies(df, columns=["track_genre"], drop_first=True)

    return df


def split_data(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )


def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
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
        "r2": r2
    }


def cross_validate_model(model, X_train, y_train, cv=5):
    neg_rmse_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    r2_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )

    cv_rmse_mean = -neg_rmse_scores.mean()
    cv_rmse_std = neg_rmse_scores.std()
    cv_r2_mean = r2_scores.mean()

    return {
        "cv_rmse_mean": cv_rmse_mean,
        "cv_rmse_std": cv_rmse_std,
        "cv_r2_mean": cv_r2_mean
    }


def compare_models(results):
    comparison_df = pd.DataFrame(results)
    numeric_cols = comparison_df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        comparison_df[col] = comparison_df[col].round(4)

    comparison_df = comparison_df.sort_values(by="rmse").reset_index(drop=True)

    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    return comparison_df


def get_feature_importance(model, X_train, top_n=10):
    importances = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    })

    importances = importances.sort_values(
        by="importance",
        ascending=False
    ).head(top_n).reset_index(drop=True)

    print("\nTop Feature Importances:")
    print(importances.to_string(index=False))

    return importances


def save_outputs(comparison_df, feature_importance_df, predictions_df, output_dir="outputs"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_path / "popularity_model_comparison.csv", index=False)
    feature_importance_df.to_csv(output_path / "popularity_feature_importance.csv", index=False)
    predictions_df.to_csv(output_path / "popularity_predictions.csv", index=False)


def run_popularity_pipeline(data_path="data/raw/spotify_data.csv", save_results=True):
    df = load_data(data_path)

    validate_data(df)
    df = handle_missing_values(df)
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    linear_model = train_linear_model(X_train, y_train)
    linear_results = evaluate_model(
        linear_model,
        X_test,
        y_test,
        "Linear Regression"
    )

    linear_cv = cross_validate_model(LinearRegression(), X_train, y_train)
    linear_results.update(linear_cv)

    random_forest_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest"
    )

    rf_cv = cross_validate_model(
        RandomForestRegressor(
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
    rf_results.update(rf_cv)

    comparison_df = compare_models([linear_results, rf_results])
    feature_importance_df = get_feature_importance(random_forest_model, X_train)

    test_predictions = pd.DataFrame({
        "actual_popularity": y_test.values,
        "linear_prediction": linear_model.predict(X_test),
        "random_forest_prediction": random_forest_model.predict(X_test)
    })

    if save_results:
        save_outputs(comparison_df, feature_importance_df, test_predictions)

    return {
        "linear_model": linear_model,
        "random_forest_model": random_forest_model,
        "comparison": comparison_df,
        "feature_importance": feature_importance_df,
        "predictions": test_predictions
    }


def main():
    run_popularity_pipeline("data/raw/spotify_data.csv")


if __name__ == "__main__":
    main()