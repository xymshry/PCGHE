"""Classification metrics and publication-ready diagnostic plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def classification_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    threshold: float = 0.0,
) -> dict[str, float | int]:
    predictions = (logits > threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()

    return {
        "n_samples": int(labels.shape[0]),
        "n_normal": int(np.sum(labels == 0)),
        "n_ventricular": int(np.sum(labels == 1)),
        "prevalence_v": float(np.mean(labels == 1)),
        "auroc": float(roc_auc_score(labels, logits)),
        "auprc": float(average_precision_score(labels, logits)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def save_plots(labels: np.ndarray, logits: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = (logits > 0.0).astype(np.int64)

    fpr, tpr, _ = roc_curve(labels, logits)
    precision, recall, _ = precision_recall_curve(labels, logits)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    axes[0].plot(fpr, tpr, color="#006d77", linewidth=1.8)
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[1].plot(recall, precision, color="#c44536", linewidth=1.8)
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_pr_curves.png", dpi=220)
    plt.close(fig)

    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(3.2, 3.0))
    image = axis.imshow(cm, cmap="Greys")
    for row in range(2):
        for col in range(2):
            axis.text(col, row, str(cm[row, col]), ha="center", va="center")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["N", "V"],
        yticklabels=["N", "V"],
        xlabel="Predicted",
        ylabel="True",
        title="Test confusion matrix",
    )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=220)
    plt.close(fig)
