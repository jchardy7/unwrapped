"""Run the descriptive summary workflow from the command line."""

from pprint import pprint

from unwrapped.io import DEFAULT_DATA_PATH
from unwrapped.summary import run_summary


def main() -> None:
    """Load the dataset, summarize it, and print key findings."""

    df, report = run_summary(DEFAULT_DATA_PATH)

    print("=== Dataset Shape ===")
    pprint(report["shape"])

    print("\n=== Numeric Column Statistics ===")
    for col, stats in report["numeric"].items():
        print(f"\n  {col}:")
        for stat, value in stats.items():
            print(f"    {stat}: {value}")

    print("\n=== Categorical Column Statistics ===")
    for col, info in report["categorical"].items():
        print(f"\n  {col} ({info['num_unique']} unique):")
        for val, count in list(info["top_values"].items())[:5]:
            print(f"    {val}: {count}")

    print("\n=== Missing Values ===")
    has_missing = {k: v for k, v in report["missing"].items() if v["count"] > 0}
    if has_missing:
        for col, info in has_missing.items():
            print(f"  {col}: {info['count']} ({info['percentage']}%)")
    else:
        print("  No missing values found.")

    print("\n=== Outliers (IQR Method) ===")
    for col, info in report["outliers"].items():
        if info["count"] > 0:
            print(f"  {col}: {info['count']} ({info['percentage']}%)")

    print("\n=== Feature Correlations with Popularity ===")
    for feature, corr in report["target_correlations"].items():
        print(f"  {feature}: {corr:+.4f}")

    print(f"\nDataset preview ({len(df)} rows):")
    print(df.head())


if __name__ == "__main__":
    main()
