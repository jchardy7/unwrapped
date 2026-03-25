# unwrapped Python Package

**About**: Final project for STAT 3250 at the University of Virginia

## How To Use This File
This is not code, it is a [Markdown](https://www.markdownguide.org/basic-syntax/) file that uses plain english to explain the project, what it does, who created it, and how a user or developer can interact with it

## Run The Demo

To run the current validation workflow on the Spotify dataset:

```bash
python scripts/demo_validation.py
```

This demo loads `data/raw/spotify_data.csv`, validates the dataset, and prints
the resulting summary report.

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
