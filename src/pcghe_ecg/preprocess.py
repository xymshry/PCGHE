"""Signal filtering and fixed-length heartbeat preprocessing."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .config import SignalConfig


def bandpass_filter(signal: np.ndarray, fs: float, config: SignalConfig) -> np.ndarray:
    """Apply a zero-phase Butterworth band-pass filter."""
    if signal.ndim != 1:
        raise ValueError("bandpass_filter expects a one-dimensional signal")
    if not 0 < config.bandpass_low_hz < config.bandpass_high_hz < fs / 2:
        raise ValueError("Band-pass frequencies must lie between 0 and Nyquist")

    sos = butter(
        config.filter_order,
        [config.bandpass_low_hz, config.bandpass_high_hz],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, signal)


def normalize_beat(beat: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Apply per-beat z-normalization without changing its length."""
    beat = np.asarray(beat, dtype=np.float64)
    return (beat - beat.mean()) / (beat.std() + epsilon)
