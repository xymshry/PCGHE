from pathlib import Path
from unittest.mock import patch

import numpy as np

from pcghe_ecg.config import SignalConfig
from pcghe_ecg.dataset import extract_record


class _RecordWithoutTargetBeats:
    fs = 360
    sig_name = ["MLII"]
    p_signal = np.zeros((1000, 1), dtype=float)


class _AnnotationsWithoutTargetBeats:
    sample = np.asarray([500])
    symbol = ["A"]


def test_record_without_n_or_v_has_two_dimensional_empty_beats():
    with (
        patch("pcghe_ecg.dataset.wfdb.rdrecord", return_value=_RecordWithoutTargetBeats()),
        patch("pcghe_ecg.dataset.wfdb.rdann", return_value=_AnnotationsWithoutTargetBeats()),
    ):
        dataset = extract_record(Path("unused"), "999", SignalConfig())

    assert dataset.beats.shape == (0, 256)
    assert dataset.labels.shape == (0,)
