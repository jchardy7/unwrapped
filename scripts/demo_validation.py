"""Run the current dataset validation workflow from the command line."""

from pprint import pprint

from unwrapped.io import DEFAULT_DATA_PATH, run_validation


def main() -> None:
    _, report = run_validation(DEFAULT_DATA_PATH)
    pprint(report)


if __name__ == "__main__":
    main()
