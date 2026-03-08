"""Error analysis for misclassified samples."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def identify_misclassified(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Identify indices and features of misclassified samples.

    Args:
        X: Feature matrix.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        feature_names: Optional feature names for DataFrame.

    Returns:
        Tuple of (misclassified_indices, X_misclassified, y_true_misclassified).
    """
    mask = y_true != y_pred
    indices = np.where(mask)[0]
    X_mis = X[indices]
    y_mis = y_true[indices]

    logger.info("Misclassified: %d / %d (%.2f%%)", len(indices), len(y_true), 100 * len(indices) / len(y_true))
    return indices, X_mis, y_mis


def compute_error_statistics(
    X_mis: np.ndarray,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute feature statistics for misclassified samples.

    Args:
        X_mis: Feature matrix of misclassified samples.
        feature_names: Optional feature names.

    Returns:
        DataFrame with mean, std, min, max per feature.
    """
    names = feature_names or [f"f{i}" for i in range(X_mis.shape[1])]
    stats = pd.DataFrame(
        {
            "mean": np.mean(X_mis, axis=0),
            "std": np.std(X_mis, axis=0),
            "min": np.min(X_mis, axis=0),
            "max": np.max(X_mis, axis=0),
        },
        index=names,
    )
    return stats


def run_error_analysis(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str] | None = None,
    output_path: Path | None = None,
) -> dict[str, float]:
    """
    Run full error analysis: identify misclassified, compute stats, optionally save.

    Args:
        X: Feature matrix.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        feature_names: Optional feature names.
        output_path: Optional path to save analysis results.

    Returns:
        Dictionary with error rate and other summary stats.
    """
    indices, X_mis, y_mis = identify_misclassified(X, y_true, y_pred, feature_names)

    results: dict[str, float] = {
        "n_misclassified": float(len(indices)),
        "n_total": float(len(y_true)),
        "error_rate": float(len(indices) / len(y_true)) if len(y_true) > 0 else 0.0,
    }

    if len(X_mis) > 0:
        stats = compute_error_statistics(X_mis, feature_names)
        logger.info("Feature statistics for misclassified samples:\n%s", stats.to_string())

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            stats.to_csv(output_path)
            logger.info("Saved error analysis to %s", output_path)

    return results
