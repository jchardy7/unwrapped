"""Run the current dataset validation workflow from the command line."""

from pprint import pprint

from unwrapped.io import DEFAULT_DATA_PATH
from unwrapped.validation import run_validation


def main() -> None:
    df, report = run_validation(DEFAULT_DATA_PATH)

    violations = report.get("range_violations", {})
    dirty = {col: n for col, n in violations.items() if n > 0}
    if dirty:
        print("Range violations detected (per column):")
        for col, n in dirty.items():
            print(f"  {col}: {n}")
        print()

    print(f"Rows: {report['num_rows']:,}  Columns: {report['num_columns']}")
    print(f"Unique tracks: {report['unique_tracks']:,}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Duplicate track_ids: {report['duplicate_track_ids']}")
    print(f"Inconsistent tracks: {report['inconsistent_tracks']}")

    print("\nFull report:")
    pprint(report)


if __name__ == "__main__":
    main()
