"""Raw and discrete-wavelet heartbeat feature extraction."""

from __future__ import annotations

import numpy as np
import pywt

from .config import FeatureConfig


def extract_beat_features(beat: np.ndarray, config: FeatureConfig) -> np.ndarray:
    beat = np.asarray(beat, dtype=np.float64)
    if beat.ndim != 1:
        raise ValueError("A beat must be a one-dimensional vector")

    if config.kind == "raw":
        return beat.copy()
    if config.kind != "dwt":
        raise ValueError(f"Unknown feature kind: {config.kind}")

    coeffs = pywt.wavedec(
        beat,
        wavelet=config.wavelet,
        level=config.wavelet_level,
        mode=config.wavelet_mode,
    )
    feature = np.concatenate(coeffs)
    if feature.shape != beat.shape:
        raise ValueError(
            "DWT feature length changed; use periodization and a power-of-two beat length"
        )
    return feature


def extract_features(beats: np.ndarray, config: FeatureConfig) -> np.ndarray:
    beats = np.asarray(beats, dtype=np.float64)
    if beats.ndim != 2:
        raise ValueError("beats must have shape (n_beats, beat_length)")
    return np.vstack([extract_beat_features(beat, config) for beat in beats])
