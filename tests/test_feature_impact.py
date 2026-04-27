import pandas as pd
import pytest

from unwrapped.feature_impact import (
    DEFAULT_SCENARIOS,
    apply_feature_changes,
    calculate_prediction_change,
    compare_feature_scenarios,
    run_feature_impact_analysis,
    save_feature_impact_results,
    simulate_feature_impact,
    validate_single_song,
)


class MockModel:
    """Simple mock model for predictable test outputs."""

    def predict(self, X):
        return X["danceability"] * 10 + X["energy"] * 5


def test_validate_single_song_accepts_one_row():
    features = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})
    validate_single_song(features)


def test_validate_single_song_rejects_empty_dataframe():
    with pytest.raises(ValueError):
        validate_single_song(pd.DataFrame())


def test_validate_single_song_rejects_multiple_rows():
    features = pd.DataFrame({
        "danceability": [0.5, 0.7],
        "energy": [0.6, 0.8],
    })

    with pytest.raises(ValueError):
        validate_single_song(features)


def test_apply_feature_changes_updates_single_feature():
    features = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})
    modified = apply_feature_changes(features, {"danceability": 0.1})

    assert modified.loc[0, "danceability"] == pytest.approx(0.6)
    assert modified.loc[0, "energy"] == pytest.approx(0.6)


def test_apply_feature_changes_updates_multiple_features():
    features = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})
    modified = apply_feature_changes(
        features,
        {"danceability": 0.1, "energy": -0.2},
    )

    assert modified.loc[0, "danceability"] == pytest.approx(0.6)
    assert modified.loc[0, "energy"] == pytest.approx(0.4)


def test_apply_feature_changes_does_not_mutate_original_dataframe():
    features = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})
    original = features.copy()

    apply_feature_changes(features, {"danceability": 0.1})

    pd.testing.assert_frame_equal(features, original)


def test_apply_feature_changes_clips_upper_bound():
    features = pd.DataFrame({"danceability": [0.95], "energy": [0.6]})
    modified = apply_feature_changes(features, {"danceability": 0.2})

    assert modified.loc[0, "danceability"] == 1.0


def test_apply_feature_changes_clips_lower_bound():
    features = pd.DataFrame({"danceability": [0.05], "energy": [0.6]})
    modified = apply_feature_changes(features, {"danceability": -0.2})

    assert modified.loc[0, "danceability"] == 0.0


@pytest.mark.parametrize(
    "feature",
    [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
    ],
)
def test_all_bounded_features_are_clipped(feature):
    features = pd.DataFrame({feature: [0.95], "tempo": [120]})
    modified = apply_feature_changes(features, {feature: 0.25})

    assert modified.loc[0, feature] == 1.0


def test_unbounded_feature_is_not_clipped():
    features = pd.DataFrame({"tempo": [120.0], "danceability": [0.5]})
    modified = apply_feature_changes(features, {"tempo": 50.0})

    assert modified.loc[0, "tempo"] == pytest.approx(170.0)


def test_apply_feature_changes_ignores_missing_feature():
    features = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})

    modified = apply_feature_changes(features, {"tempo": 10})

    pd.testing.assert_frame_equal(features, modified)


def test_calculate_prediction_change_positive_change():
    result = calculate_prediction_change(50, 60)

    assert result["absolute_change"] == 10
    assert result["percent_change"] == 20


def test_calculate_prediction_change_negative_change():
    result = calculate_prediction_change(80, 60)

    assert result["absolute_change"] == -20
    assert result["percent_change"] == -25


def test_calculate_prediction_change_zero_original():
    result = calculate_prediction_change(0, 10)

    assert result["absolute_change"] == 10
    assert result["percent_change"] == 0


def test_calculate_prediction_change_rounds_values():
    result = calculate_prediction_change(33, 40)

    assert result["absolute_change"] == 7
    assert result["percent_change"] == pytest.approx(21.2121)


def test_simulate_feature_impact_returns_expected_values():
    model = MockModel()
    features = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})

    result = simulate_feature_impact(
        model,
        features,
        {"danceability": 0.1},
    )

    assert result["original_prediction"] == pytest.approx(8.0)
    assert result["modified_prediction"] == pytest.approx(9.0)
    assert result["absolute_change"] == pytest.approx(1.0)
    assert result["percent_change"] == pytest.approx(12.5)


def test_simulate_feature_impact_with_multiple_feature_changes():
    model = MockModel()
    features = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})

    result = simulate_feature_impact(
        model,
        features,
        {"danceability": 0.1, "energy": 0.1},
    )

    assert result["modified_prediction"] > result["original_prediction"]
    assert result["absolute_change"] == pytest.approx(1.5)


def test_simulate_feature_impact_rejects_empty_features():
    model = MockModel()

    with pytest.raises(ValueError):
        simulate_feature_impact(model, pd.DataFrame(), {"danceability": 0.1})


def test_simulate_feature_impact_rejects_multiple_rows():
    model = MockModel()
    features = pd.DataFrame({
        "danceability": [0.5, 0.6],
        "energy": [0.6, 0.7],
    })

    with pytest.raises(ValueError):
        simulate_feature_impact(model, features, {"danceability": 0.1})


def test_compare_feature_scenarios_returns_dataframe():
    model = MockModel()
    features = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})

    scenarios = {
        "danceability_boost": {"danceability": 0.1},
        "energy_boost": {"energy": 0.1},
    }

    result = compare_feature_scenarios(model, features, scenarios)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2


def test_compare_feature_scenarios_returns_expected_columns():
    model = MockModel()
    features = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})

    result = compare_feature_scenarios(
        model,
        features,
        {"danceability_boost": {"danceability": 0.1}},
    )

    expected_columns = {
        "scenario",
        "changes",
        "original_prediction",
        "modified_prediction",
        "absolute_change",
        "percent_change",
        "rank",
    }

    assert expected_columns.issubset(set(result.columns))


def test_compare_feature_scenarios_ranks_larger_impact_first():
    model = MockModel()
    features = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})

    scenarios = {
        "small_danceability_boost": {"danceability": 0.1},
        "large_danceability_boost": {"danceability": 0.2},
    }

    result = compare_feature_scenarios(model, features, scenarios)

    assert result.loc[0, "scenario"] == "large_danceability_boost"
    assert result.loc[0, "rank"] == 1


def test_compare_feature_scenarios_uses_default_scenarios():
    model = MockModel()
    features = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})

    result = compare_feature_scenarios(model, features)

    assert len(result) == len(DEFAULT_SCENARIOS)


def test_save_feature_impact_results_creates_csv(tmp_path):
    results = pd.DataFrame({
        "scenario": ["increase_danceability"],
        "absolute_change": [1.5],
    })

    output_file = save_feature_impact_results(results, output_dir=tmp_path)

    saved = pd.read_csv(output_file)

    assert "feature_impact_results.csv" in output_file
    pd.testing.assert_frame_equal(saved, results)


def test_run_feature_impact_analysis_rejects_negative_song_index(monkeypatch):
    def mock_prepare_feature_impact_model(data_path):
        model = MockModel()
        X_test = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})
        y_test = pd.Series([50])
        return model, X_test, y_test

    monkeypatch.setattr(
        "unwrapped.feature_impact.prepare_feature_impact_model",
        mock_prepare_feature_impact_model,
    )

    with pytest.raises(ValueError):
        run_feature_impact_analysis(song_index=-1, save_results=False)


def test_run_feature_impact_analysis_rejects_out_of_range_index(monkeypatch):
    def mock_prepare_feature_impact_model(data_path):
        model = MockModel()
        X_test = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})
        y_test = pd.Series([50])
        return model, X_test, y_test

    monkeypatch.setattr(
        "unwrapped.feature_impact.prepare_feature_impact_model",
        mock_prepare_feature_impact_model,
    )

    with pytest.raises(ValueError):
        run_feature_impact_analysis(song_index=5, save_results=False)


def test_run_feature_impact_analysis_returns_results(monkeypatch):
    def mock_prepare_feature_impact_model(data_path):
        model = MockModel()
        X_test = pd.DataFrame({"danceability": [0.5], "energy": [0.6]})
        y_test = pd.Series([50])
        return model, X_test, y_test

    monkeypatch.setattr(
        "unwrapped.feature_impact.prepare_feature_impact_model",
        mock_prepare_feature_impact_model,
    )

    result = run_feature_impact_analysis(song_index=0, save_results=False)

    assert isinstance(result, pd.DataFrame)
    assert "actual_popularity" in result.columns
    assert result["actual_popularity"].iloc[0] == 50