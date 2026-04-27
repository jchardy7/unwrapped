"""Shared constants for the unwrapped package."""

from __future__ import annotations

AUDIO_FEATURES: list[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

AUDIO_FEATURES_WITH_DURATION: list[str] = AUDIO_FEATURES + [
    "duration_ms",
]

AUDIO_FEATURES_FOR_HIT_CLASSIFICATION: list[str] = AUDIO_FEATURES + [
    "duration_ms",
    "explicit",
    "key",
    "mode",
    "time_signature",
]