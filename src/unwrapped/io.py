# example of a module we will use for the project; this module will cover all the input/output (hence, i.o.), so the functions we will write to load in the raw data

import pandas as pd

EXPECTED_COLUMNS = {
    'track_id', 'artists', 'album_name', 'track_name',
    'popularity', 'duration_ms', 'explicit',
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', 'time_signature',
    'track_genre'
}

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    return df


def validate_schema(df: pd.DataFrame):
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    extra = set(df.columns) - EXPECTED_COLUMNS
    if extra:
        print(f"[Warning] Extra columns detected: {extra}")


def validate_ranges(df: pd.DataFrame):
    errors = []

    def check_range(col, low, high):
        if not df[col].between(low, high).all():
            errors.append(f"{col} out of range [{low}, {high}]")

    check_range('danceability', 0, 1)
    check_range('energy', 0, 1)
    check_range('valence', 0, 1)
    check_range('acousticness', 0, 1)
    check_range('instrumentalness', 0, 1)
    check_range('speechiness', 0, 1)

    if (df['tempo'] <= 0).any():
        errors.append("tempo contains non-positive values")

    if (df['duration_ms'] <= 0).any():
        errors.append("duration_ms contains non-positive values")

    if errors:
        raise ValueError("Range validation failed:\n" + "\n".join(errors))


def validate_duplicates(df: pd.DataFrame):
    duplicate_rows = df.duplicated().sum()
    duplicate_tracks = df['track_id'].duplicated().sum()

    return {
        "duplicate_rows": int(duplicate_rows),
        "duplicate_track_ids": int(duplicate_tracks)
    }


def validate_track_consistency(df: pd.DataFrame):
    grouped = df.groupby('track_id')[
        ['danceability', 'energy', 'tempo', 'duration_ms']
    ].nunique()

    inconsistent_tracks = grouped[(grouped > 1).any(axis=1)]

    return int(len(inconsistent_tracks))


def validate_correlations(df: pd.DataFrame):
    corr = df[['energy', 'loudness']].corr().iloc[0, 1]

    if corr < 0.5:
        print(f"[Warning] Weak correlation between energy and loudness: {corr:.2f}")


def missing_summary(df: pd.DataFrame):
    return df.isnull().sum().to_dict()


def validation_report(df: pd.DataFrame) -> dict:
    report = {}

    report['num_rows'] = int(len(df))
    report['num_columns'] = int(len(df.columns))

    report['missing_values'] = missing_summary(df)

    report.update(validate_duplicates(df))

    report['unique_tracks'] = int(df['track_id'].nunique())

    report['inconsistent_tracks'] = validate_track_consistency(df)

    return report


def run_validation(path: str):
    df = load_data(path)

    validate_schema(df)
    validate_ranges(df)
    validate_correlations(df)

    report = validation_report(df)

    return df, report


if __name__ == "__main__":
    path = "spotify_data.csv"

    df, report = run_validation(path)

    print("\nValidation Report:")
    for key, value in report.items():
        print(f"{key}: {value}")