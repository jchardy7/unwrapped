import pandas as pd
import pytest

from unwrapped.hit_shape_predictor import (
    validate_data,
    handle_missing_values,
    preprocess_data,
    create_hit_label,
    build_hit_profiles,
    calculate_profile_differences,
    compute_similarity_features,
    build_modeling_dataframe,
    split_data,
    train_logistic_model,
    train_random_forest,
    evaluate_model,
    compare_models,
    build_predictions_df,
)


def make_sample_df():
    """Create a small dataframe that mimics the Spotify dataset."""
    data = {
        "Unnamed: 0": [0, 1, 2, 3, 4, 5, 6, 7],
        "track_id": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "artists": ["x", "y", "z", "w", "p", "q", "r", "s"],
        "album_name": ["al1", "al2", "al3", "al4", "al5", "al6", "al7", "al8"],
        "track_name": ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8"],
        "track_genre": ["pop", "pop", "rock", "rock", "hip-hop", "jazz", "pop", "rock"],
        "popularity": [80, 75, 20, 30, 90, 15, 68, 72],
        "duration_ms": [200000, 210000, 180000, 190000, 220000, 175000, 205000, 215000],
        "explicit": [0, 0, 1, 0, 1, 0, 0, 1],
        "danceability": [0.70, 0.65, 0.40, 0.45, 0.75, 0.35, 0.62, 0.68],
        "energy": [0.80, 0.75, 0.50, 0.55, 0.85, 0.45, 0.70, 0.78],
        "key": [5, 6, 4, 3, 7, 2, 5, 6],
        "loudness": [-5, -6, -10, -9, -4, -11, -7, -5],
        "mode": [1, 1, 0, 1, 1, 0, 1, 1],
        "speechiness": [0.05, 0.06, 0.04, 0.03, 0.07, 0.02, 0.05, 0.06],
        "acousticness": [0.10, 0.15, 0.60, 0.50, 0.08, 0.70, 0.20, 0.12],
        "instrumentalness": [0.00, 0.00, 0.20, 0.30, 0.00, 0.35, 0.05, 0.01],
        "liveness": [0.10, 0.12, 0.20, 0.18, 0.11, 0.25, 0.14, 0.13],
        "valence": [0.60, 0.65, 0.40, 0.35, 0.72, 0.30, 0.58, 0.66],
        "tempo": [120, 118, 100, 105, 125, 95, 116, 121],
        "time_signature": [4, 4, 4, 4, 4, 4, 4, 4],
    }
    return pd.DataFrame(data)


def prepare_df():
    df = make_sample_df()
    validate_data(df)
    df = handle_missing_values(df)
    df = preprocess_data(df)
    df = create_hit_label(df, threshold=70)
    return df


def test_validate_data_passes_with_valid_dataframe():
    df = make_sample_df()
    validate_data(df)


def test_validate_data_raises_for_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(ValueError):
        validate_data(df)


def test_validate_data_raises_for_missing_required_column():
    df = make_sample_df().drop(columns=["tempo"])
    with pytest.raises(ValueError):
        validate_data(df)


def test_handle_missing_values_drops_missing_popularity():
    df = make_sample_df()
    df.loc[0, "popularity"] = None

    cleaned = handle_missing_values(df)

    assert len(cleaned) == 7
    assert cleaned["popularity"].isna().sum() == 0


def test_handle_missing_values_fills_numeric_missing_values():
    df = make_sample_df()
    df.loc[0, "danceability"] = None
    df.loc[1, "energy"] = None

    cleaned = handle_missing_values(df)

    assert cleaned["danceability"].isna().sum() == 0
    assert cleaned["energy"].isna().sum() == 0


def test_preprocess_data_drops_unused_columns():
    df = make_sample_df()
    processed = preprocess_data(df)

    assert "Unnamed: 0" not in processed.columns
    assert "track_id" not in processed.columns
    assert "artists" not in processed.columns
    assert "album_name" not in processed.columns
    assert "track_name" not in processed.columns
    assert "track_genre" not in processed.columns


def test_create_hit_label_adds_binary_column():
    df = preprocess_data(handle_missing_values(make_sample_df()))
    df = create_hit_label(df, threshold=70)

    assert "is_hit" in df.columns
    assert set(df["is_hit"].unique()) == {0, 1}


def test_create_hit_label_counts_hits_correctly():
    df = preprocess_data(handle_missing_values(make_sample_df()))
    df = create_hit_label(df, threshold=70)

    assert df["is_hit"].sum() == 4


def test_build_hit_profiles_returns_two_groups():
    df = prepare_df()
    profiles = build_hit_profiles(df)

    assert profiles.shape[0] == 2
    assert 0 in profiles.index
    assert 1 in profiles.index


def test_build_hit_profiles_contains_expected_features():
    df = prepare_df()
    profiles = build_hit_profiles(df)

    assert "danceability" in profiles.columns
    assert "energy" in profiles.columns
    assert "tempo" in profiles.columns


def test_build_hit_profiles_raises_if_only_one_class_present():
    df = prepare_df()
    df["is_hit"] = 1

    with pytest.raises(ValueError):
        build_hit_profiles(df)


def test_calculate_profile_differences_returns_expected_columns():
    df = prepare_df()
    profiles = build_hit_profiles(df)
    diff_df = calculate_profile_differences(profiles)

    assert "feature" in diff_df.columns
    assert "non_hit_mean" in diff_df.columns
    assert "hit_mean" in diff_df.columns
    assert "difference_hit_minus_non_hit" in diff_df.columns
    assert "absolute_difference" in diff_df.columns


def test_compute_similarity_features_adds_engineered_columns():
    df = prepare_df()
    similarity_df = compute_similarity_features(df)

    assert "distance_to_non_hit" in similarity_df.columns
    assert "distance_to_hit" in similarity_df.columns
    assert "hit_distance_advantage" in similarity_df.columns
    assert "hit_closeness_ratio" in similarity_df.columns


def test_build_modeling_dataframe_has_expected_columns_only():
    df = prepare_df()
    similarity_df = compute_similarity_features(df)
    model_df = build_modeling_dataframe(similarity_df)

    expected_columns = {
        "distance_to_non_hit",
        "distance_to_hit",
        "hit_distance_advantage",
        "hit_closeness_ratio",
        "is_hit",
    }

    assert set(model_df.columns) == expected_columns


def test_split_data_returns_nonempty_sets():
    df = prepare_df()
    similarity_df = compute_similarity_features(df)
    model_df = build_modeling_dataframe(similarity_df)

    X_train, X_test, y_train, y_test = split_data(
        model_df, test_size=0.25, random_state=42
    )

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_train_logistic_model_has_predict_method():
    df = prepare_df()
    similarity_df = compute_similarity_features(df)
    model_df = build_modeling_dataframe(similarity_df)
    X_train, X_test, y_train, y_test = split_data(
        model_df, test_size=0.25, random_state=42
    )

    model = train_logistic_model(X_train, y_train)
    assert hasattr(model, "predict")


def test_train_random_forest_has_feature_importances():
    df = prepare_df()
    similarity_df = compute_similarity_features(df)
    model_df = build_modeling_dataframe(similarity_df)
    X_train, X_test, y_train, y_test = split_data(
        model_df, test_size=0.25, random_state=42
    )

    model = train_random_forest(X_train, y_train)
    assert hasattr(model, "feature_importances_")


def test_evaluate_model_returns_expected_metric_keys():
    df = prepare_df()
    similarity_df = compute_similarity_features(df)
    model_df = build_modeling_dataframe(similarity_df)
    X_train, X_test, y_train, y_test = split_data(
        model_df, test_size=0.25, random_state=42
    )

    model = train_logistic_model(X_train, y_train)
    results = evaluate_model(model, X_test, y_test, "Logistic Regression")

    assert "model" in results
    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results


def test_compare_models_sorts_highest_f1_first():
    results = [
        {"model": "A", "accuracy": 0.80, "precision": 0.70, "recall": 0.60, "f1": 0.65},
        {"model": "B", "accuracy": 0.82, "precision": 0.75, "recall": 0.70, "f1": 0.72},
    ]

    comparison = compare_models(results)

    assert comparison.iloc[0]["model"] == "B"


def test_build_predictions_df_has_expected_columns():
    df = prepare_df()
    similarity_df = compute_similarity_features(df)
    model_df = build_modeling_dataframe(similarity_df)
    X_train, X_test, y_train, y_test = split_data(
        model_df, test_size=0.25, random_state=42
    )

    model = train_random_forest(X_train, y_train)
    predictions_df = build_predictions_df(model, X_test, y_test)

    assert "actual_is_hit" in predictions_df.columns
    assert "predicted_is_hit" in predictions_df.columns
    assert len(predictions_df) == len(y_test)