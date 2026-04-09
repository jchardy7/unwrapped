import pandas as pd
from unwrapped.summary import summarize_dataset

df = pd.DataFrame({
    "artist_name": ["A", "B", "A"],
    "genre": ["pop", "rock", "pop"],
    "popularity": [50, 60, 70]
})

summary = summarize_dataset(df)

for key, value in summary.items():
    print(f"{key}: {value}")