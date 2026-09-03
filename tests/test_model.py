from pathlib import Path

import numpy as np

from pcghe_ecg.config import ModelConfig
from pcghe_ecg.model import PlaintextModel


def test_exported_linear_logit_matches_sklearn(tmp_path: Path):
    rng = np.random.default_rng(17)
    features = rng.normal(size=(120, 32))
    labels = (features[:, 0] - 0.5 * features[:, 1] > 0).astype(int)

    model = PlaintextModel(ModelConfig(pca_components=8)).fit(features, labels)
    assert model.verify_manual_logit(features) < 1e-10

    model.save(tmp_path)
    with np.load(tmp_path / "ckks_interface.npz") as artifact:
        transformed = model.transform(features)
        exported_logit = transformed @ artifact["weight"] + artifact["bias"][0]
    assert np.allclose(exported_logit, model.decision_function(features))
