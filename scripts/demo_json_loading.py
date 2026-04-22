"""Demonstrate loading a Spotify dataset sample via the JSON loader."""

import json
import tempfile
from pathlib import Path

from unwrapped.io import DEFAULT_DATA_PATH, load_data, load_json


def main() -> None:
    """Load the CSV, save a sample as JSON, reload it, and print a preview."""

    print("Loading CSV dataset...")
    df = load_data(DEFAULT_DATA_PATH)

    sample = df.head(10)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(json.dumps(sample.to_dict(orient="records")))

    print(f"\nSaved 10-row sample to temporary JSON file: {tmp_path.name}")

    loaded_df = load_json(str(tmp_path))

    print(f"\nLoaded JSON shape: {loaded_df.shape}")
    print("\nColumns:", list(loaded_df.columns))
    print("\nData preview:")
    print(loaded_df.head())

    tmp_path.unlink()
    print("\nTemporary file cleaned up.")

if __name__ == "__main__":
    main()


