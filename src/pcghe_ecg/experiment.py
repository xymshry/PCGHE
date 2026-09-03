"""End-to-end plaintext experiment orchestration and artifact capture."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import scipy
import sklearn
import wfdb

from .config import (
    DS2_RECORDS,
    TRAIN_RECORDS,
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
)
from .dataset import BeatDataset, load_or_build_dataset
from .evaluation import classification_metrics, save_plots
from .features import extract_features
from .model import PlaintextModel


def _split_dataset(dataset: BeatDataset) -> dict[str, BeatDataset]:
    result = {
        "train": dataset.select_records(TRAIN_RECORDS),
        "test": dataset.select_records(DS2_RECORDS),
    }
    for name, split in result.items():
        classes = np.unique(split.labels)
        if not np.array_equal(classes, np.asarray([0, 1])):
            raise ValueError(f"Split {name} does not contain both N and V classes")
    return result


def _prediction_frame(split: BeatDataset, logits: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": split.record_ids,
            "sample_index": split.sample_indices,
            "symbol": split.symbols,
            "label": split.labels,
            "logit": logits,
            "prediction": (logits > 0.0).astype(np.int64),
        }
    )


def _software_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pywavelets": pywt.__version__,
        "scikit_learn": sklearn.__version__,
        "wfdb": wfdb.__version__,
    }


def run_experiment(
    data_dir: Path,
    output_dir: Path,
    cache_file: Path,
    config: ExperimentConfig,
    force_cache: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_or_build_dataset(
        data_dir=data_dir,
        cache_file=cache_file,
        config=config.signal,
        force=force_cache,
    )
    splits = _split_dataset(dataset)

    feature_matrices = {
        name: extract_features(split.beats, config.feature)
        for name, split in splits.items()
    }

    model = PlaintextModel(config.model).fit(
        feature_matrices["train"], splits["train"].labels
    )
    manual_logit_error = model.verify_manual_logit(feature_matrices["test"])

    metrics: dict[str, dict] = {}
    logits: dict[str, np.ndarray] = {}
    for name in ("test",):
        logits[name] = model.decision_function(feature_matrices[name])
        metrics[name] = classification_metrics(
            splits[name].labels,
            logits[name],
            threshold=config.model.threshold,
        )

    class_counts = []
    for name, split in splits.items():
        for symbol, label in (("N", 0), ("V", 1)):
            class_counts.append(
                {
                    "split": name,
                    "symbol": symbol,
                    "count": int(np.sum(split.labels == label)),
                    "records": int(np.unique(split.record_ids[split.labels == label]).shape[0]),
                }
            )
    pd.DataFrame(class_counts).to_csv(output_dir / "class_counts.csv", index=False)

    for name in ("test",):
        _prediction_frame(splits[name], logits[name]).to_csv(
            output_dir / f"predictions_{name}.csv", index=False
        )

    save_plots(splits["test"].labels, logits["test"], output_dir)
    artifact_dir = output_dir / "artifacts"
    model.save(artifact_dir)
    np.savez_compressed(
        artifact_dir / "ckks_test_vectors.npz",
        features=model.transform(feature_matrices["test"]),
        labels=splits["test"].labels,
        record_ids=splits["test"].record_ids,
        sample_indices=splits["test"].sample_indices,
        plaintext_logits=logits["test"],
    )

    run_config = config.to_dict()
    run_config["software"] = _software_versions()
    run_config["manual_logit_max_abs_error"] = manual_logit_error
    (output_dir / "config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(metrics).T.rename_axis("split").to_csv(output_dir / "metrics.csv")
    return metrics


def run_baselines(
    data_dir: Path,
    output_dir: Path,
    cache_file: Path,
    force_cache: bool = False,
) -> pd.DataFrame:
    configurations = (
        ("raw_no_pca", "raw", None),
        ("raw_pca32", "raw", 32),
        ("dwt_no_pca", "dwt", None),
        ("dwt_pca32", "dwt", 32),
    )
    rows = []
    for index, (name, feature_kind, components) in enumerate(configurations):
        config = ExperimentConfig(
            feature=FeatureConfig(kind=feature_kind),
            model=ModelConfig(pca_components=components),
        )
        metrics = run_experiment(
            data_dir,
            output_dir / name,
            cache_file,
            config,
            force_cache=force_cache and index == 0,
        )
        rows.append(
            {
                "configuration": name,
                "feature": feature_kind,
                "pca_components": components if components is not None else "none",
                **metrics["test"],
            }
        )
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "baseline_summary.csv", index=False)
    return frame


def run_dimension_sweep(
    data_dir: Path,
    output_dir: Path,
    cache_file: Path,
    dimensions: tuple[int, ...] = (8, 16, 32, 64, 128, 256),
    force_cache: bool = False,
) -> pd.DataFrame:
    rows = []
    for index, dimension in enumerate(dimensions):
        config = ExperimentConfig(model=ModelConfig(pca_components=dimension))
        metrics = run_experiment(
            data_dir,
            output_dir / f"pca_{dimension}",
            cache_file,
            config,
            force_cache=force_cache and index == 0,
        )
        rows.append({"pca_components": dimension, **metrics["test"]})
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "dimension_sweep.csv", index=False)
    return frame
