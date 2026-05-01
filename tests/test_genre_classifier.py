from __future__ import annotations

from pathlib import Path
from typing import Any


import numpy as np
import pandas as pd
import pytest


from unwrapped.genre_classifier import (
    compare_genre_models,
    confusion_matrix_df,
    evaluate_genre_model,
    prepare_genre_data,
    prepare_genre_train_test_data,
    run_genre_classifier_pipeline,
    save_outputs,
    split_genre_data,
    top_k_accuracy,
    train_logistic_genre_classifier,
    train_random_forest_genre_classifier,
    validate_data,
)


def make_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "track_id": "t",
        "artists": "Artist",
        "album_name": "Album",
        "track_name": "Song",
        "track_genre": "pop",
        "popularity": 50,
        "duration_ms": 200000,
        "explicit": 0,
        "danceability": 0.6,
        "energy": 0.7,
        "key": 5,
        "loudness": -5.0,
        "mode": 1,
        "speechiness": 0.05,
        "acousticness": 0.2,
        "instrumentalness": 0.0,
        "liveness": 0.1,
        "valence": 0.5,
        "tempo": 120.0,
        "time_signature": 4,
    }
    row.update(overrides)
    return row


def make_df() -> pd.DataFrame:
    """Three genres with 8 rows each — enough for a 0.25 stratified split."""
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(0)
    for i in range(8):
        rows.append(
            make_row(
                track_id=f"pop-{i}",
                track_genre="pop",
                danceability=0.7 + rng.normal(0, 0.02),
                energy=0.8 + rng.normal(0, 0.02),
                valence=0.7 + rng.normal(0, 0.02),
                acousticness=0.1 + rng.normal(0, 0.02),
                tempo=120 + rng.normal(0, 1),
            )
        )
    for i in range(8):
        rows.append(
            make_row(
                track_id=f"rock-{i}",
                track_genre="rock",
                danceability=0.4 + rng.normal(0, 0.02),
                energy=0.85 + rng.normal(0, 0.02),
                valence=0.4 + rng.normal(0, 0.02),
                acousticness=0.05 + rng.normal(0, 0.02),
                tempo=140 + rng.normal(0, 1),
            )
        )
    for i in range(8):
        rows.append(
            make_row(
                track_id=f"acoustic-{i}",
                track_genre="acoustic",
                danceability=0.3 + rng.normal(0, 0.02),
                energy=0.2 + rng.normal(0, 0.02),
                valence=0.5 + rng.normal(0, 0.02),
                acousticness=0.85 + rng.normal(0, 0.02),
                tempo=90 + rng.normal(0, 1),
            )
        )
    return pd.DataFrame(rows)


def test_validate_data_passes_for_valid_dataframe():
    validate_data(make_df())


def test_validate_data_raises_on_empty_dataframe():
    with pytest.raises(ValueError, match="empty"):
        validate_data(pd.DataFrame())


def test_validate_data_raises_when_track_genre_missing():
    df = make_df().drop(columns=["track_genre"])
    with pytest.raises(ValueError, match="track_genre"):
        validate_data(df)


def test_prepare_genre_data_filters_rare_genres():
    df = make_df()
    df = pd.concat([df, pd.DataFrame([make_row(track_genre="metal")])], ignore_index=True)

    X, y = prepare_genre_data(df, min_samples_per_genre=5)

    assert "metal" not in set(y)
    assert set(y) == {"pop", "rock", "acoustic"}
    assert len(X) == len(y) == 24


def test_prepare_genre_data_raises_when_all_genres_filtered_out():
    df = make_df()
    with pytest.raises(ValueError, match="No genres"):
        prepare_genre_data(df, min_samples_per_genre=999)


def test_prepare_genre_data_returns_audio_feature_columns():
    df = make_df()
    X, _ = prepare_genre_data(df, min_samples_per_genre=5)

    expected = {
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
        "explicit", "key", "mode", "time_signature",
    }
    assert expected <= set(X.columns)
    assert "track_genre" not in X.columns


def test_prepare_genre_train_test_data_uses_training_medians(
    monkeypatch: pytest.MonkeyPatch,
):
    df = pd.DataFrame(
        [
            make_row(track_id="pop-train-missing", track_genre="pop", danceability=None),
            make_row(track_id="pop-train", track_genre="pop", danceability=0.2),
            make_row(track_id="rock-train", track_genre="rock", danceability=0.4),
            make_row(track_id="pop-test-missing", track_genre="pop", danceability=None),
            make_row(track_id="rock-test", track_genre="rock", danceability=0.95),
        ]
    )

    def fake_split(frame, *args, **kwargs):
        train_ids = {"pop-train-missing", "pop-train", "rock-train"}
        train = frame[frame["track_id"].isin(train_ids)]
        test = frame[~frame["track_id"].isin(train_ids)]
        return train, test

    monkeypatch.setattr("unwrapped.genre_classifier.train_test_split", fake_split)

    X_train, X_test, _, _ = prepare_genre_train_test_data(
        df,
        min_samples_per_genre=1,
    )

    assert X_train.loc[0, "danceability"] == pytest.approx(0.3)
    assert X_test.loc[3, "danceability"] == pytest.approx(0.3)


def test_split_genre_data_preserves_class_set():
    df = make_df()
    X, y = prepare_genre_data(df, min_samples_per_genre=5)
    X_train, X_test, y_train, y_test = split_genre_data(X, y, test_size=0.25)

    assert len(X_train) > 0 and len(X_test) > 0
    assert set(y_train) == set(y_test) == set(y)


def test_train_logistic_returns_pipeline_with_predict_proba():
    df = make_df()
    X, y = prepare_genre_data(df, min_samples_per_genre=5)
    X_train, _, y_train, _ = split_genre_data(X, y, test_size=0.25)

    model = train_logistic_genre_classifier(X_train, y_train)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")


def test_train_random_forest_has_feature_importances():
    df = make_df()
    X, y = prepare_genre_data(df, min_samples_per_genre=5)
    X_train, _, y_train, _ = split_genre_data(X, y, test_size=0.25)

    model = train_random_forest_genre_classifier(X_train, y_train)
    assert hasattr(model, "feature_importances_")
    assert len(model.feature_importances_) == X_train.shape[1]


def test_top_k_accuracy_within_unit_interval_and_at_least_accuracy():
    df = make_df()
    X, y = prepare_genre_data(df, min_samples_per_genre=5)
    X_train, X_test, y_train, y_test = split_genre_data(X, y, test_size=0.25)
    model = train_random_forest_genre_classifier(X_train, y_train)

    top1 = top_k_accuracy(model, X_test, y_test, k=1)
    top3 = top_k_accuracy(model, X_test, y_test, k=3)

    assert 0.0 <= top1 <= 1.0
    assert 0.0 <= top3 <= 1.0
    assert top3 >= top1 - 1e-9


def test_top_k_accuracy_requires_predict_proba():
    class NoProba:
        classes_ = np.array(["a"])

        def predict(self, X):
            return ["a"] * len(X)

    with pytest.raises(TypeError, match="predict_proba"):
        top_k_accuracy(NoProba(), pd.DataFrame({"x": [1]}), pd.Series(["a"]))


def test_evaluate_genre_model_returns_expected_keys():
    df = make_df()
    X, y = prepare_genre_data(df, min_samples_per_genre=5)
    X_train, X_test, y_train, y_test = split_genre_data(X, y, test_size=0.25)
    model = train_random_forest_genre_classifier(X_train, y_train)

    result = evaluate_genre_model(model, X_test, y_test, "RF", top_k=2)

    assert set(result.keys()) == {
        "model", "accuracy", "top_k_accuracy", "macro_f1", "weighted_f1"
    }
    assert result["model"] == "RF"


def test_compare_genre_models_sorts_by_macro_f1_desc():
    results = [
        {"model": "A", "accuracy": 0.8, "top_k_accuracy": 0.95, "macro_f1": 0.6, "weighted_f1": 0.7},
        {"model": "B", "accuracy": 0.7, "top_k_accuracy": 0.90, "macro_f1": 0.8, "weighted_f1": 0.75},
    ]

    comparison = compare_genre_models(results)

    assert comparison.iloc[0]["model"] == "B"
    assert comparison["macro_f1"].is_monotonic_decreasing


def test_confusion_matrix_df_is_square_with_class_labels():
    df = make_df()
    X, y = prepare_genre_data(df, min_samples_per_genre=5)
    X_train, X_test, y_train, y_test = split_genre_data(X, y, test_size=0.25)
    model = train_random_forest_genre_classifier(X_train, y_train)

    cm = confusion_matrix_df(model, X_test, y_test)

    assert cm.shape[0] == cm.shape[1]
    assert set(cm.index) == set(cm.columns)
    assert set(y_test).issubset(set(cm.index))


def test_save_outputs_writes_three_csvs(tmp_path: Path):
    comparison = pd.DataFrame({"model": ["A"], "macro_f1": [0.8]})
    confusion = pd.DataFrame([[1, 0], [0, 1]], index=["a", "b"], columns=["a", "b"])
    predictions = pd.DataFrame({"actual_genre": ["a"], "random_forest_prediction": ["a"]})

    paths = save_outputs(comparison, confusion, predictions, output_dir=tmp_path / "out")

    assert set(paths.keys()) == {"comparison", "confusion_matrix", "predictions"}
    assert (tmp_path / "out" / "genre_model_comparison.csv").exists()
    assert (tmp_path / "out" / "genre_confusion_matrix.csv").exists()
    assert (tmp_path / "out" / "genre_predictions.csv").exists()


def test_run_genre_classifier_pipeline_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    big_df = pd.concat([make_df()] * 3, ignore_index=True)
    big_df["track_id"] = [str(i) for i in range(len(big_df))]

    monkeypatch.setattr("unwrapped.genre_classifier.load_data", lambda _: big_df)

    output_dir = tmp_path / "pipeline"
    result = run_genre_classifier_pipeline(
        data_path="fake.csv",
        min_samples_per_genre=5,
        save_results=True,
        output_dir=output_dir,
    )

    assert {"logistic_model", "random_forest_model", "comparison", "confusion_matrix",
            "predictions"} <= result.keys()
    assert not result["comparison"].empty
    assert (output_dir / "genre_model_comparison.csv").exists()
