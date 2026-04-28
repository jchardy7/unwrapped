"""Utilities for loading the Spotify dataset and persisting trained models."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

DEFAULT_DATA_PATH = "data/raw/spotify_data.csv"

def _drop_index_column(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the Unnamed: 0 export index column if present."""
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Spotify CSV into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file to load.

    Returns
    -------
    pd.DataFrame
        Raw Spotify dataset with the common export index column removed when
        present.
    """

    df = pd.read_csv(path)
    return _drop_index_column(df)


def load_json(path: str) -> pd.DataFrame:
    """Load a Spotify JSON export into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the JSON file to load.

    Returns
    -------
    pd.DataFrame
        Raw Spotify dataset with the common export index column removed when
        present.
    """
    df = pd.read_json(path)
    return _drop_index_column(df)


def _meta_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def save_model(
    model: Any,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Persist a fitted estimator to disk along with a JSON metadata sidecar.

    The estimator is serialized with ``joblib.dump`` and a sibling
    ``<path>.meta.json`` file records the model class, sklearn version, the
    training timestamp, and any caller-supplied keys (e.g. ``feature_names``,
    ``target``).

    Parameters
    ----------
    model : Any
        A fitted scikit-learn estimator (or any joblib-serializable object).
    path : str | Path
        Destination file path. Parent directories are created when missing.
    metadata : dict, optional
        Extra keys merged into the sidecar JSON.

    Returns
    -------
    dict
        Mapping with ``"model"`` and ``"metadata"`` keys pointing to the
        written file paths.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)

    sidecar: dict[str, Any] = {
        "model_class": type(model).__name__,
        "sklearn_version": sklearn.__version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        sidecar.update(metadata)

    meta_path = _meta_path(path)
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=2, default=str)

    return {"model": str(path), "metadata": str(meta_path)}


def load_model(path: str | Path) -> tuple[Any, dict[str, Any] | None]:
    """Load a model previously written by :func:`save_model`.

    Returns ``(model, metadata)``. ``metadata`` is ``None`` when the sidecar
    file is absent, so this helper still works on bare joblib dumps.
    """
    path = Path(path)
    model = joblib.load(path)

    meta_path = _meta_path(path)
    metadata: dict[str, Any] | None = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

    return model, metadata

