"""Run the current dataset cleaning workflow from the command line."""

from pprint import pprint

from unwrapped.io import DEFAULT_DATA_PATH
from unwrapped.clean import run_cleaning


def main() -> None:
    """Load the raw dataset, clean it, and print a short demo summary."""

    cleaned_df, report = run_cleaning(DEFAULT_DATA_PATH)

    print("Cleaning report:")
    pprint(report)

    print("\nCleaned data preview:")
    print(cleaned_df.head())


if __name__ == "__main__":
    main()
