"""
Group 3: Popularity Model Comparison

Models:
- Linear Regression
- Random Forest
- CatBoost

Outputs:
- Model performance table
- Actual vs predicted plot for the best model
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor


# --------------------------
# CONFIG
# --------------------------

DATA_PATHS = [
    Path("/Users/norahmasrour/PyCharmMiscProject/spotify_final.csv"),
    Path("/Users/norahmasrour/PyCharmMiscProject/spotify_data.csv"),
    Path("data/raw/spotify_final.csv"),
    Path("data/raw/spotify_data.csv"),
    Path("spotify_final.csv"),
    Path("spotify_data.csv"),
]

OUTPUT_DIR = Path("outputs")

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
    "duration_ms",
]


# --------------------------
# HELPERS
# --------------------------

def find_dataset():
    for path in DATA_PATHS:
        if path.exists():
            return path

    raise FileNotFoundError("Could not find Spotify dataset.")


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return df


def get_genre_col(df):
    if "track_genre" in df.columns:
        return "track_genre"
    if "genre" in df.columns:
        return "genre"
    return None


def prepare_data(df):
    if "popularity" not in df.columns:
        raise ValueError("Dataset must contain a popularity column.")

    numeric = [c for c in AUDIO_FEATURES if c in df.columns]
    genre_col = get_genre_col(df)

    features = numeric + ([genre_col] if genre_col else [])

    df = df[features + ["popularity"]].dropna()
    df = df[df["popularity"] > 0]

    return df[features], df["popularity"], numeric, genre_col


def evaluate(name, y_true, y_pred):
    return {
        "model": name,
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 3),
        "mae": round(mean_absolute_error(y_true, y_pred), 3),
        "r2": round(r2_score(y_true, y_pred), 3),
    }


def build_preprocessor(numeric, genre, scale=False):
    transformers = []

    if numeric:
        transformers.append(
            ("num", StandardScaler() if scale else "passthrough", numeric)
        )

    if genre:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), [genre])
        )

    return ColumnTransformer(transformers=transformers)


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    OUTPUT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.35)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()])
    plt.xlabel("Actual Popularity")
    plt.ylabel("Predicted Popularity")
    plt.title(f"Actual vs Predicted Popularity: {model_name}")
    plt.tight_layout()

    output_path = OUTPUT_DIR / "group3_actual_vs_predicted.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved plot to: {output_path}")


# --------------------------
# MODELS
# --------------------------

def run_linear(X_train, X_test, y_train, y_test, numeric, genre):
    print("Training Linear Regression...")
    start = time.time()

    model = Pipeline([
        ("prep", build_preprocessor(numeric, genre, True)),
        ("model", LinearRegression())
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"Finished Linear Regression in {round(time.time() - start, 2)} seconds")

    return evaluate("Linear Regression", y_test, preds), model, preds


def run_random_forest(X_train, X_test, y_train, y_test, numeric, genre):
    print("Training Random Forest...")
    start = time.time()

    model = Pipeline([
        ("prep", build_preprocessor(numeric, genre, False)),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"Finished Random Forest in {round(time.time() - start, 2)} seconds")

    return evaluate("Random Forest", y_test, preds), model, preds


def run_catboost(X_train, X_test, y_train, y_test, genre):
    print("Training CatBoost...")
    start = time.time()

    cat_features = []
    if genre:
        cat_features = [X_train.columns.get_loc(genre)]

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.08,
        depth=6,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    model.fit(X_train, y_train, cat_features=cat_features)
    preds = model.predict(X_test)

    print(f"Finished CatBoost in {round(time.time() - start, 2)} seconds")

    return evaluate("CatBoost", y_test, preds), model, preds


# --------------------------
# MAIN
# --------------------------

def main():
    path = find_dataset()
    print(f"Using dataset: {path}")

    df = load_data(path)
    X, y, numeric, genre = prepare_data(df)

    print(f"Rows used: {len(X)}")
    print(f"Numeric features: {numeric}")
    print(f"Genre column: {genre}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    all_results = []

    linear_result, linear_model, linear_preds = run_linear(
        X_train, X_test, y_train, y_test, numeric, genre
    )
    all_results.append((linear_result, linear_model, linear_preds))

    rf_result, rf_model, rf_preds = run_random_forest(
        X_train, X_test, y_train, y_test, numeric, genre
    )
    all_results.append((rf_result, rf_model, rf_preds))

    cat_result, cat_model, cat_preds = run_catboost(
        X_train, X_test, y_train, y_test, genre
    )
    all_results.append((cat_result, cat_model, cat_preds))

    results_df = pd.DataFrame([item[0] for item in all_results])
    results_df = results_df.sort_values("rmse").reset_index(drop=True)

    print("\nModel Results:")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model"]
    print(f"\nBest Model: {best_model_name}")

    for result, model, preds in all_results:
        if result["model"] == best_model_name:
            plot_actual_vs_predicted(y_test, preds, best_model_name)
            break


if __name__ == "__main__":
    main()