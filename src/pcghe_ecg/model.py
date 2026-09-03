"""Training-only transforms and a CKKS-compatible linear classifier."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import ModelConfig


class PlaintextModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.pca: PCA | None = None
        if config.pca_components is not None:
            self.pca = PCA(
                n_components=config.pca_components,
                svd_solver="full",
                random_state=config.random_state,
            )
        self.classifier = LogisticRegression(
            class_weight=config.class_weight,
            C=config.c,
            max_iter=config.max_iter,
            solver="lbfgs",
            random_state=config.random_state,
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "PlaintextModel":
        scaled = self.scaler.fit_transform(features)
        transformed = self.pca.fit_transform(scaled) if self.pca else scaled
        self.classifier.fit(transformed, labels)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        scaled = self.scaler.transform(features)
        return self.pca.transform(scaled) if self.pca else scaled

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        return self.classifier.decision_function(self.transform(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.decision_function(features) > self.config.threshold).astype(np.int64)

    @property
    def weight(self) -> np.ndarray:
        return self.classifier.coef_[0].copy()

    @property
    def bias(self) -> float:
        return float(self.classifier.intercept_[0] - self.config.threshold)

    def verify_manual_logit(self, features: np.ndarray, tolerance: float = 1e-10) -> float:
        transformed = self.transform(features)
        manual = transformed @ self.weight + float(self.classifier.intercept_[0])
        library = self.classifier.decision_function(transformed)
        maximum_error = float(np.max(np.abs(manual - library)))
        if maximum_error > tolerance:
            raise AssertionError(
                f"Manual and library logits differ by {maximum_error:.3e}"
            )
        return maximum_error

    def save(self, artifact_dir: Path) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, artifact_dir / "plaintext_pipeline.joblib")

        if self.pca is None:
            pca_mean = np.zeros(self.scaler.mean_.shape[0], dtype=np.float64)
            pca_components = np.eye(self.scaler.mean_.shape[0], dtype=np.float64)
            pca_variance_ratio = np.ones(self.scaler.mean_.shape[0], dtype=np.float64)
        else:
            pca_mean = self.pca.mean_
            pca_components = self.pca.components_
            pca_variance_ratio = self.pca.explained_variance_ratio_

        np.savez(
            artifact_dir / "ckks_interface.npz",
            scaler_mean=self.scaler.mean_,
            scaler_scale=self.scaler.scale_,
            pca_mean=pca_mean,
            pca_components=pca_components,
            pca_explained_variance_ratio=pca_variance_ratio,
            weight=self.weight,
            bias=np.asarray([self.bias], dtype=np.float64),
            threshold=np.asarray([self.config.threshold], dtype=np.float64),
        )

        metadata = {
            "model_config": asdict(self.config),
            "plaintext_feature_dimension": int(self.weight.shape[0]),
            "ckks_server_operation": "logit = dot(encrypted_feature, weight) + bias",
            "decision_rule": "class V if decrypted logit > 0, otherwise class N",
        }
        (artifact_dir / "artifact_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
