"""
Popularity regression module for predicting Spotify track popularity
using audio features and metadata.

This module validates the input data, handles missing values, compares
a baseline linear regression model to a random forest regressor, and
reports feature importance for the random forest.
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# Columns required for the modeling pipeline.
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


# Load the Spotify dataset from a CSV file.
def load_data(path):
    return pd.read_csv(path)


# Validate that the dataframe contains the columns needed for the model.
def validate_data(df):
    if df.empty:
        raise ValueError("Input dataframe is empty.")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )


# Handle missing values before modeling.
# - Drop rows missing the target (popularity)
# - Fill missing genre values with 'unknown'
# - Fill missing numeric feature values with the column median
def handle_missing_values(df):
    df = df.copy()

    df = df.dropna(subset=["popularity"])

    df["track_genre"] = df["track_genre"].fillna("unknown")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_feature_cols = [col for col in numeric_cols if col != "popularity"]

    for col in numeric_feature_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


# Remove identifier columns and one-hot encode genre for modeling.
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


# Split the dataset into features/target and then into train/test sets.
def split_data(df):
    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# Train a baseline linear regression model.
def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# Train a random forest regression model to capture nonlinear patterns.
def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


# Evaluate a model using RMSE and R², then print and return the results.
def evaluate_model(model, X_test, y_test, model_name="Model"):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"{model_name} RMSE: {rmse:.2f}")
    print(f"{model_name} R2: {r2:.4f}")

    return {"model": model_name, "rmse": rmse, "r2": r2}


# Create a summary table comparing model performance.
def compare_models(results):
    comparison_df = pd.DataFrame(results)
    comparison_df["rmse"] = comparison_df["rmse"].round(2)
    comparison_df["r2"] = comparison_df["r2"].round(4)

    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))

    return comparison_df


# Show the most important predictors in the random forest model.
def show_feature_importance(model, X_train, top_n=10):
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    importances = importances.sort_values(ascending=False).head(top_n)

    print("\nTop Feature Importances (relative contribution to Random Forest predictions):")
    for feature, importance in importances.items():
        print(f"{feature}: {importance:.4f} ({importance * 100:.2f}% of model importance)")

    return importances


# Run the full popularity prediction workflow end-to-end.
def main():
    df = load_data("data/raw/spotify_data.csv")

    validate_data(df)
    df = handle_missing_values(df)
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    linear_model = train_linear_model(X_train, y_train)
    linear_results = evaluate_model(linear_model, X_test, y_test, "Linear Regression")

    random_forest_model = train_random_forest(X_train, y_train)
    rf_results = evaluate_model(random_forest_model, X_test, y_test, "Random Forest")

    compare_models([linear_results, rf_results])
    show_feature_importance(random_forest_model, X_train)


# Execute the script when run directly.
if __name__ == "__main__":
    main()