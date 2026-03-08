"""Gradient boosting classifiers for phishing detection."""

import logging
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

ClassifierType = Literal["xgboost", "lightgbm", "catboost"]


def _get_classifier(
    model_type: ClassifierType,
    learning_rate: float = 0.0,
    n_estimators: int = 100,
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """Instantiate classifier by type."""
    if model_type == "xgboost":
        import xgboost as xgb

        return xgb.XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=random_state,
            eval_metric="logloss",
            **kwargs,
        )
    if model_type == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            random_state=random_state,
            verbose=-1,
            **kwargs,
        )
    if model_type == "catboost":
        import catboost as cb

        return cb.CatBoostClassifier(
            learning_rate=learning_rate,
            iterations=n_estimators,
            random_state=random_state,
            verbose=0,
            **kwargs,
        )
    raise ValueError(f"Unknown model: {model_type}. Use xgboost, lightgbm, or catboost.")


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: ClassifierType = "xgboost",
    learning_rate: float = 0.01,
    n_estimators: int = 100,
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """
    Train a gradient boosting classifier.

    Args:
        X_train: Training features.
        y_train: Training labels.
        model_type: One of 'xgboost', 'lightgbm', 'catboost'.
        learning_rate: Learning rate.
        n_estimators: Number of boosting rounds.
        random_state: Random seed.
        **kwargs: Additional classifier arguments.

    Returns:
        Fitted classifier.

    Raises:
        ValueError: If model_type is invalid.
    """
    clf = _get_classifier(
        model_type=model_type,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        random_state=random_state,
        **kwargs,
    )

    clf.fit(X_train, y_train)
    logger.info("Trained %s classifier (n_estimators=%d, lr=%.4f)", model_type, n_estimators, learning_rate)
    return clf


def predict(
    model: Any,
    X: np.ndarray,
    model_type: ClassifierType | None = None,
) -> np.ndarray:
    """
    Predict class labels.

    Args:
        model: Fitted classifier.
        X: Feature matrix.
        model_type: Optional; used for logging only.

    Returns:
        Predicted labels.

    Raises:
        ValueError: If model is None.
    """
    if model is None:
        raise ValueError("Model is None. Train first.")
    return np.asarray(model.predict(X), dtype=np.int64)


def predict_proba(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """
    Predict class probabilities.

    Args:
        model: Fitted classifier.
        X: Feature matrix.

    Returns:
        Probability array of shape (n_samples, n_classes).
    """
    if model is None:
        raise ValueError("Model is None. Train first.")
    return np.asarray(model.predict_proba(X), dtype=np.float64)
