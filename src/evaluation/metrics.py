"""Evaluation metrics for phishing detection."""

import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    average: str = "binary",
) -> dict[str, float]:
    """
    Compute F1, precision, recall, ROC-AUC, and accuracy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class (optional, for ROC-AUC).
        average: Averaging for multi-class ('binary', 'macro', 'weighted').

    Returns:
        Dictionary with metric names and values.
    """
    metrics: dict[str, float] = {
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    if y_prob is not None:
        try:
            if y_prob.ndim == 2:
                y_prob = y_prob[:, 1]
            # ROC-AUC: 'binary' is invalid; use 'macro' for binary (equivalent)
            roc_average = "macro" if average == "binary" else average
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob, average=roc_average))
        except ValueError as e:
            logger.warning("ROC-AUC could not be computed: %s", e)
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def print_metrics(metrics: dict[str, float]) -> None:
    """Print metrics in a readable format."""
    for name, value in metrics.items():
        logger.info("%s: %.4f", name, value)
