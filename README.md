# unwrapped Python Package

**About**: Final project for STAT 3250 at the University of Virginia

## How To Use This File
This is not code, it is a [Markdown](https://www.markdownguide.org/basic-syntax/) file that uses plain english to explain the project, what it does, who created it, and how a user or developer can interact with it

## Project Goals

This package helps our STAT 3250 team explore a Spotify tracks dataset in a
reproducible way. Right now the project includes a loading layer, a cleaning
workflow, and a validation workflow so the dataset can be prepared for later
analysis and modeling.

## Setup

Install the package and development tools with:

```bash
pip install -e ".[dev]"
```

This installs the core package dependencies listed in
`pyproject.toml`, including pandas and numpy, plus pytest for testing.

## Run The Demos

To run the cleaning workflow demo on the Spotify dataset:

```bash
python scripts/demo_cleaning.py
```

This demo loads `data/raw/spotify_data.csv`, applies the cleaning pipeline,
prints the cleaning summary report, and shows a preview of the cleaned data.

To run the validation workflow demo on the Spotify dataset:

```bash
python scripts/demo_validation.py
```

This demo loads `data/raw/spotify_data.csv`, validates the dataset, and prints
the resulting summary report.

## How Cleaning Works

The cleaning workflow runs in a fixed sequence:

1. The dataset is loaded from CSV into a pandas DataFrame.
2. If the file contains the extra `Unnamed: 0` column, it is removed.
3. Text columns are stripped of extra whitespace, and blank strings are turned
   into missing values.
4. The `explicit` column is normalized into boolean values.
5. Numeric analysis columns are coerced to numeric types with invalid strings
   converted to missing values.
6. Rows missing required identifiers such as `track_id`, `artists`,
   `track_name`, and `track_genre` are dropped.
7. Rows with invalid core numeric values, such as non-positive `tempo` or
   `duration_ms`, or out-of-range audio features, are removed.
8. Exact duplicate rows are dropped, and repeated `track_id` values are reduced
   to one canonical row using completeness and popularity as tie-breakers.

The cleaning entrypoint returns both the cleaned DataFrame and a report that
summarizes what changed at each stage.

## How Validation Works

The validation workflow runs in a fixed sequence:

1. The dataset is loaded from CSV into a pandas DataFrame.
2. If the file contains the extra `Unnamed: 0` column, it is dropped during
   loading.
3. The schema check verifies that all expected Spotify fields are present.
   Missing required columns raise a `ValueError`, while unexpected extra
   columns produce a warning.
4. Range checks validate selected numeric fields. `danceability`, `energy`,
   `valence`, `acousticness`, `instrumentalness`, and `speechiness` must fall
   between `0` and `1`. `tempo` and `duration_ms` must be positive. Any
   failure raises a `ValueError`.
5. A correlation check compares `energy` and `loudness`. If their correlation
   is below `0.5`, the workflow prints a warning instead of failing.
6. After validation passes, the report summarizes row and column counts,
   missing values by column, duplicate rows, duplicate `track_id` values, the
   number of unique tracks, and track IDs whose core audio features are
   inconsistent across repeated rows.

The validation entrypoint returns both the loaded DataFrame and the summary
report.
