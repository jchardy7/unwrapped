"""Feature Impact Analyzer for Spotify popularity predictions.

This module runs counterfactual-style simulations by changing selected
audio features and measuring how those changes affect predicted popularity.

It builds on the existing popularity pipeline by reusing the same validation,
missing-value handling, preprocessing, train/test split, and random forest
training logic from ``popularity.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io import load_data
from .popularity import (
    DEFAULT_OUTPUT_DIR,
    handle_missing_values,
    preprocess_data,
    split_data,
    train_random_forest,
    validate_data,
)


BOUNDED_FEATURES: dict[str, tuple[float, float]] = {
    "danceability": (0.0, 1.0),
    "energy": (0.0, 1.0),
    "speechiness": (0.0, 1.0),
    "acousticness": (0.0, 1.0),
    "instrumentalness": (0.0, 1.0),
    "liveness": (0.0, 1.0),
    "valence": (0.0, 1.0),
}


DEFAULT_SCENARIOS: dict[str, dict[str, float]] = {
    "increase_danceability": {"danceability": 0.10},
    "increase_energy": {"energy": 0.10},
    "increase_valence": {"valence": 0.10},
    "decrease_acousticness": {"acousticness": -0.10},
    "higher_energy_and_danceability": {
        "energy": 0.10,
        "danceability": 0.10,
    },
}


def validate_single_song(features: pd.DataFrame) -> None:
    """Validate that the feature input contains exactly one song."""
    if features.empty:
        raise ValueError("Input features cannot be empty.")

    if len(features) != 1:
        raise ValueError("Feature Impact Analyzer expects exactly one song.")


def apply_feature_changes(
    features: pd.DataFrame,
    changes: dict[str, float],
) -> pd.DataFrame:
    """Apply feature changes to a copy of the original song features."""
    modified_features = features.copy()

    for feature, change in changes.items():
        if feature not in modified_features.columns:
            continue

        modified_features[feature] = modified_features[feature] + change

        if feature in BOUNDED_FEATURES:
            lower_bound, upper_bound = BOUNDED_FEATURES[feature]
            modified_features[feature] = modified_features[feature].clip(
                lower=lower_bound,
                upper=upper_bound,
            )

    return modified_features


def calculate_prediction_change(
    original_prediction: float,
    modified_prediction: float,
) -> dict[str, float]:
    """Calculate absolute and percent prediction changes using NumPy."""
    absolute_change = np.subtract(modified_prediction, original_prediction)

    if original_prediction != 0:
        percent_change = np.multiply(
            np.divide(absolute_change, original_prediction),
            100,
        )
    else:
        percent_change = 0.0

    return {
        "absolute_change": round(float(absolute_change), 4),
        "percent_change": round(float(percent_change), 4),
    }


def simulate_feature_impact(
    model: Any,
    base_features: pd.DataFrame,
    changes: dict[str, float],
) -> dict[str, float]:
    """Estimate how selected feature changes affect predicted popularity."""
    validate_single_song(base_features)

    original_prediction = float(model.predict(base_features)[0])
    modified_features = apply_feature_changes(base_features, changes)
    modified_prediction = float(model.predict(modified_features)[0])

    change_summary = calculate_prediction_change(
        original_prediction,
        modified_prediction,
    )

    return {
        "original_prediction": round(original_prediction, 4),
        "modified_prediction": round(modified_prediction, 4),
        **change_summary,
    }


def compare_feature_scenarios(
    model: Any,
    base_features: pd.DataFrame,
    scenarios: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Compare several feature-change scenarios for one song."""
    validate_single_song(base_features)

    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    results = []

    for scenario_name, changes in scenarios.items():
        impact = simulate_feature_impact(model, base_features, changes)

        results.append({
            "scenario": scenario_name,
            "changes": str(changes),
            **impact,
        })

    scenario_df = pd.DataFrame(results)

    scenario_df["rank"] = (
        scenario_df["absolute_change"]
        .rank(ascending=False, method="dense")
        .astype(int)
    )

    return scenario_df.sort_values(
        by="absolute_change",
        ascending=False,
    ).reset_index(drop=True)


def prepare_feature_impact_model(
    data_path: str = "data/raw/spotify_data.csv",
) -> tuple[Any, pd.DataFrame, pd.Series]:
    """Train the popularity model and return test features for simulation."""
    df = load_data(data_path)

    validate_data(df)
    df = handle_missing_values(df)
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)
    model = train_random_forest(X_train, y_train)

    return model, X_test, y_test


def save_feature_impact_results(
    results_df: pd.DataFrame,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> str:
    """Save Feature Impact Analyzer results to a CSV file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_path = output_path / "feature_impact_results.csv"
    results_df.to_csv(results_path, index=False)

    return str(results_path)


def run_feature_impact_analysis(
    data_path: str = "data/raw/spotify_data.csv",
    song_index: int = 0,
    save_results: bool = True,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> pd.DataFrame:
    """Run the full Feature Impact Analyzer workflow."""
    model, X_test, y_test = prepare_feature_impact_model(data_path)

    if song_index < 0 or song_index >= len(X_test):
        raise ValueError(
            f"song_index must be between 0 and {len(X_test) - 1}."
        )

    base_song = X_test.iloc[[song_index]]
    actual_popularity = float(y_test.iloc[song_index])

    results_df = compare_feature_scenarios(model, base_song)
    results_df.insert(0, "song_index", song_index)
    results_df.insert(1, "actual_popularity", actual_popularity)

    if save_results:
        save_feature_impact_results(results_df, output_dir)

    print("\nFeature Impact Analyzer Results:")
    print(results_df.to_string(index=False))

    return results_df


def main() -> None:
    """Run the Feature Impact Analyzer from the command line."""
    data_path = "data/raw/spotify_data.csv"

    try:
        run_feature_impact_analysis(data_path=data_path)
    except FileNotFoundError:
        print(
            f"Error: data file not found at '{data_path}'.\n"
            "Make sure you are running this from the repository root."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()