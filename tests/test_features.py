import numpy as np

from pcghe_ecg.config import FeatureConfig
from pcghe_ecg.features import extract_features


def test_dwt_periodization_preserves_feature_length():
    rng = np.random.default_rng(7)
    beats = rng.normal(size=(5, 256))
    features = extract_features(beats, FeatureConfig(kind="dwt"))
    assert features.shape == (5, 256)
    assert np.isfinite(features).all()


def test_raw_features_are_copied():
    beats = np.arange(512, dtype=float).reshape(2, 256)
    features = extract_features(beats, FeatureConfig(kind="raw"))
    assert np.array_equal(features, beats)
    assert not np.shares_memory(features, beats)
