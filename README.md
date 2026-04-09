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

## Preference Scoring Tool

The preference tool lets you build a personalized "taste profile" from a list of songs you like, then scores every track in the dataset by how closely it matches that profile.

**How it works:** It computes the mean audio feature vector across all your liked songs (your taste profile), then ranks every other song in the dataset by cosine similarity to that profile. Scores are normalized to a 0–1 range, where 1 is the closest match to your taste.

### Usage

```python
from unwrapped.io import load_tracks
from unwrapped.preference import LikedSongs

df = load_tracks("path/to/spotify_data.csv")

# Create a LikedSongs object
liked = LikedSongs(df)

# Add songs by Spotify track ID or by name
liked.add_by_id("5SuOikwiRyPMVoIQDJUgSV")
liked.add_by_name("Blinding Lights")
liked.add_by_name("Levitating", artist="Dua Lipa")  # use artist to disambiguate

# See what's in your liked list
liked.show()

# Predict preference scores for all songs
scores = liked.predict(top_n=20)
print(scores)
```

### Output

`predict()` returns a DataFrame sorted by `preference_score` descending:

| track_id | track_name | artists | track_genre | popularity | preference_score |
|---|---|---|---|---|---|
| 3n3Ppam7vgaVa1iaRUIOKE | Flowers | Miley Cyrus | pop | 87 | 0.9741 |
| ... | ... | ... | ... | ... | ... |

### Saving and loading your liked list

Your liked songs list can be saved to a file and reloaded in a future session, so you don't have to rebuild it every time:

```python
# Save
liked.save("my_likes.json")

# Load in a future session
liked = LikedSongs(df)
liked.load("my_likes.json")
```

### Parameters

**`LikedSongs(df)`**
- `df` — a DataFrame loaded via `unwrapped.io.load_tracks()`

**`add_by_name(track_name, artist=None)`**
- `artist` is optional but recommended when multiple songs share the same title

**`predict(top_n=None, exclude_liked=True)`**
- `top_n` — return only the N highest-scoring songs (default: all songs)
- `exclude_liked` — if `True`, your liked songs are excluded from results (default: `True`)
