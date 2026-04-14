from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from unwrapped.popularity import (
    compare_models,
    cross_validate_model,
    evaluate_model,
    get_feature_importance,
    handle_missing_values,
    preprocess_data,
    run_popularity_pipeline,
    save_outputs,
    split_data,
    train_linear_model,
    train_random_forest,
    validate_data,
)


# Reusable valid sample dataframe for testing the popularity pipeline.
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "track_id": [str(i) for i in range(1, 13)],
        "track_name": [f"track_{i}" for i in range(1, 13)],
        "album_name": [f"album_{i}" for i in range(1, 13)],
        "artists": [f"artist_{i}" for i in range(1, 13)],
        "Unnamed: 0": list(range(12)),
        "track_genre": [
            "pop", "rock", "jazz", "pop", "rock", "jazz",
            "pop", "rock", "jazz", "pop", "rock", "jazz"
        ],
        "popularity": [50, 60, 70, 80, 90, 55, 65, 75, 85, 95, 58, 68],
        "duration_ms": [200000, 210000, 220000, 230000, 240000, 205000, 215000, 225000, 235000, 245000, 208000, 218000],
        "danceability": [0.80, 0.60, 0.70, 0.50, 0.90, 0.82, 0.62, 0.72, 0.52, 0.92, 0.78, 0.68],
        "energy": [0.70, 0.50, 0.80, 0.40, 0.90, 0.72, 0.52, 0.82, 0.42, 0.92, 0.68, 0.58],
        "loudness": [-5.0, -6.0, -4.5, -7.0, -3.5, -5.2, -6.2, -4.7, -7.2, -3.2, -5.5, -6.1],
        "speechiness": [0.05, 0.04, 0.06, 0.03, 0.07, 0.05, 0.04, 0.06, 0.03, 0.07, 0.05, 0.04],
        "acousticness": [0.10, 0.20, 0.15, 0.30, 0.05, 0.12, 0.22, 0.17, 0.32, 0.07, 0.11, 0.21],
        "instrumentalness": [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        "liveness": [0.10, 0.20, 0.15, 0.30, 0.12, 0.11, 0.21, 0.16, 0.31, 0.13, 0.10, 0.20],
        "valence": [0.60, 0.50, 0.70, 0.40, 0.80, 0.62, 0.52, 0.72, 0.42, 0.82, 0.58, 0.48],
        "tempo": [120.0, 110.0, 130.0, 100.0, 140.0, 121.0, 111.0, 131.0, 101.0, 141.0, 119.0, 109.0],
    })


# Validation should succeed for a properly structured dataframe.
def test_validate_data_accepts_valid_dataframe(sample_df):
    validate_data(sample_df)


# Validation should fail on an empty dataframe.
def test_validate_data_raises_on_empty_dataframe():
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Input dataframe is empty"):
        validate_data(empty_df)


# Validation should fail when required columns are missing.
def test_validate_data_raises_on_missing_columns(sample_df):
    bad_df = sample_df.drop(columns=["tempo", "track_genre"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(bad_df)


# Missing popularity rows should be dropped and missing predictors filled.
def test_handle_missing_values_fills_and_drops():
    df = pd.DataFrame({
        "track_genre": ["pop", None, "rock"],
        "popularity": [50, None, 70],
        "duration_ms": [200000, 210000, None],
        "danceability": [0.8, None, 0.7],
        "energy": [0.7, 0.5, None],
        "loudness": [-5.0, -6.0, None],
        "speechiness": [0.05, None, 0.06],
        "acousticness": [0.10, None, 0.15],
        "instrumentalness": [0.0, None, 0.0],
        "liveness": [0.10, None, 0.15],
        "valence": [0.6, None, 0.7],
        "tempo": [120.0, None, 130.0],
    })

    cleaned = handle_missing_values(df)

    assert cleaned["popularity"].isna().sum() == 0
    assert cleaned["track_genre"].isna().sum() == 0
    assert cleaned.isna().sum().sum() == 0
    assert len(cleaned) == 2


# If all popularity values are missing, the cleaned dataframe should be empty.
def test_handle_missing_values_all_missing_target():
    df = pd.DataFrame({
        "track_genre": ["pop", "rock"],
        "popularity": [None, None],
        "duration_ms": [200000, 210000],
        "danceability": [0.8, 0.6],
        "energy": [0.7, 0.5],
        "loudness": [-5.0, -6.0],
        "speechiness": [0.05, 0.04],
        "acousticness": [0.10, 0.20],
        "instrumentalness": [0.0, 0.0],
        "liveness": [0.10, 0.20],
        "valence": [0.6, 0.5],
        "tempo": [120.0, 110.0],
    })

    cleaned = handle_missing_values(df)

    assert cleaned.empty


# Missing-value handling should not mutate the original dataframe.
def test_handle_missing_values_does_not_modify_original():
    df = pd.DataFrame({
        "track_genre": ["pop", None],
        "popularity": [50, 60],
        "duration_ms": [200000, None],
        "danceability": [0.8, None],
        "energy": [0.7, None],
        "loudness": [-5.0, None],
        "speechiness": [0.05, None],
        "acousticness": [0.10, None],
        "instrumentalness": [0.0, None],
        "liveness": [0.10, None],
        "valence": [0.6, None],
        "tempo": [120.0, None],
    })

    original = df.copy(deep=True)
    handle_missing_values(df)

    pd.testing.assert_frame_equal(df, original)


# Preprocessing should remove identifier columns and encode genre dummies.
def test_preprocess_data_drops_ids_and_encodes_genre(sample_df):
    processed = preprocess_data(sample_df)

    assert "track_id" not in processed.columns
    assert "track_name" not in processed.columns
    assert "album_name" not in processed.columns
    assert "artists" not in processed.columns
    assert "Unnamed: 0" not in processed.columns
    assert "track_genre" not in processed.columns
    assert any(col.startswith("track_genre_") for col in processed.columns)


# Preprocessing should not mutate the original dataframe.
def test_preprocess_data_does_not_modify_original(sample_df):
    original = sample_df.copy(deep=True)
    preprocess_data(sample_df)

    pd.testing.assert_frame_equal(sample_df, original)


# Train/test splitting should return non-empty sets and keep the target out of X.
def test_split_data_returns_train_test_sets(sample_df):
    processed = preprocess_data(sample_df)
    X_train, X_test, y_train, y_test = split_data(processed)

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert "popularity" not in X_train.columns


# Linear regression should fit and return numeric evaluation results.
def test_train_linear_model_and_evaluate(sample_df):
    processed = preprocess_data(sample_df)
    X_train, X_test, y_train, y_test = split_data(processed)

    model = train_linear_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test, "Linear Regression")

    assert results["model"] == "Linear Regression"
    assert isinstance(results["rmse"], float)
    assert isinstance(results["mae"], float)
    assert isinstance(results["r2"], float)


# Random forest should fit and return numeric evaluation results.
def test_train_random_forest_and_evaluate(sample_df):
    processed = preprocess_data(sample_df)
    X_train, X_test, y_train, y_test = split_data(processed)

    model = train_random_forest(X_train, y_train)
    results = evaluate_model(model, X_test, y_test, "Random Forest")

    assert results["model"] == "Random Forest"
    assert isinstance(results["rmse"], float)
    assert isinstance(results["mae"], float)
    assert isinstance(results["r2"], float)


# Model comparison should return a tidy summary dataframe sorted by RMSE.
def test_compare_models_returns_dataframe():
    results = [
        {"model": "Linear Regression", "rmse": 19.12, "mae": 15.0, "r2": 0.2589},
        {"model": "Random Forest", "rmse": 15.29, "mae": 12.0, "r2": 0.5264},
    ]

    comparison = compare_models(results)

    assert "model" in comparison.columns
    assert "rmse" in comparison.columns
    assert "mae" in comparison.columns
    assert "r2" in comparison.columns
    assert len(comparison) == 2
    # Should be sorted by rmse ascending
    assert comparison.iloc[0]["rmse"] <= comparison.iloc[1]["rmse"]


# Model comparison should round all numeric metrics to 4 decimal places.
def test_compare_models_rounds_metrics():
    results = [
        {"model": "Linear Regression", "rmse": 19.123456, "mae": 15.12345, "r2": 0.258912},
    ]

    comparison = compare_models(results)

    assert comparison.loc[0, "rmse"] == 19.1235
    assert comparison.loc[0, "mae"] == 15.1234
    assert comparison.loc[0, "r2"] == 0.2589


# Feature importance should return a DataFrame of the requested length.
def test_get_feature_importance_returns_dataframe(sample_df):
    processed = preprocess_data(sample_df)
    X_train, X_test, y_train, y_test = split_data(processed)

    model = train_random_forest(X_train, y_train)
    importances = get_feature_importance(model, X_train, top_n=5)

    assert isinstance(importances, pd.DataFrame)
    assert len(importances) == 5
    assert "feature" in importances.columns
    assert "importance" in importances.columns


# Feature importances should be sorted from highest to lowest.
def test_get_feature_importance_sorted_desc(sample_df):
    processed = preprocess_data(sample_df)
    X_train, X_test, y_train, y_test = split_data(processed)

    model = train_random_forest(X_train, y_train)
    importances = get_feature_importance(model, X_train, top_n=5)

    assert importances["importance"].is_monotonic_decreasing


# Cross-validation should return RMSE and R^2 summary statistics.
def test_cross_validate_model_returns_expected_keys(sample_df):
    processed = preprocess_data(sample_df)
    X_train, _, y_train, _ = split_data(processed, test_size=0.25)

    cv_results = cross_validate_model(LinearRegression(), X_train, y_train, cv=3)

    assert set(cv_results.keys()) == {"cv_rmse_mean", "cv_rmse_std", "cv_r2_mean"}
    assert isinstance(cv_results["cv_rmse_mean"], float)
    assert cv_results["cv_rmse_mean"] >= 0
    assert cv_results["cv_rmse_std"] >= 0


# save_outputs should write three CSVs into the requested directory.
def test_save_outputs_writes_three_csvs(tmp_path: Path):
    comparison = pd.DataFrame({"model": ["A", "B"], "rmse": [1.0, 2.0]})
    importance = pd.DataFrame({"feature": ["x", "y"], "importance": [0.6, 0.4]})
    predictions = pd.DataFrame(
        {"actual_popularity": [50, 60], "linear_prediction": [49.0, 61.0]}
    )

    output_dir = tmp_path / "nested" / "outputs"
    paths = save_outputs(comparison, importance, predictions, output_dir=output_dir)

    assert set(paths.keys()) == {"comparison", "feature_importance", "predictions"}
    assert (output_dir / "popularity_model_comparison.csv").exists()
    assert (output_dir / "popularity_feature_importance.csv").exists()
    assert (output_dir / "popularity_predictions.csv").exists()

    round_trip = pd.read_csv(output_dir / "popularity_model_comparison.csv")
    assert list(round_trip.columns) == ["model", "rmse"]


# run_popularity_pipeline should call load_data, train both models, and save outputs.
def test_run_popularity_pipeline_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_df
):
    # Expand the fixture with more rows so cross-validation has enough data.
    big_df = pd.concat([sample_df] * 6, ignore_index=True)
    big_df["track_id"] = [str(i) for i in range(len(big_df))]

    def fake_load(path: str) -> pd.DataFrame:
        assert path == "fake.csv"
        return big_df

    monkeypatch.setattr("unwrapped.popularity.load_data", fake_load)

    output_dir = tmp_path / "pipeline_outputs"
    result = run_popularity_pipeline(
        data_path="fake.csv",
        save_results=True,
        output_dir=output_dir,
    )

    assert {"linear_model", "random_forest_model", "comparison",
            "feature_importance", "predictions"} <= result.keys()
    assert not result["comparison"].empty
    assert not result["feature_importance"].empty
    assert (output_dir / "popularity_model_comparison.csv").exists()
    assert (output_dir / "popularity_feature_importance.csv").exists()
    assert (output_dir / "popularity_predictions.csv").exists()


# Passing save_results=False should skip writing files.
def test_run_popularity_pipeline_skips_saving_when_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, sample_df
):
    big_df = pd.concat([sample_df] * 6, ignore_index=True)
    big_df["track_id"] = [str(i) for i in range(len(big_df))]

    monkeypatch.setattr("unwrapped.popularity.load_data", lambda _: big_df)

    output_dir = tmp_path / "should_not_exist"
    run_popularity_pipeline(
        data_path="fake.csv",
        save_results=False,
        output_dir=output_dir,
    )

    assert not output_dir.exists()
