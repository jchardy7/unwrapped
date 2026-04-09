import pandas as pd
from unwrapped.summary import summarize_dataset


def test_summary_rows():
    df = pd.DataFrame({
        "artist_name": ["A", "B", "A"],
        "genre": ["pop", "rock", "pop"],
        "popularity": [50, 60, 70]
    })

    result = summarize_dataset(df)

    assert result["n_rows"] == 3
    assert result["unique_artists"] == 2
    assert result["unique_genres"] == 2