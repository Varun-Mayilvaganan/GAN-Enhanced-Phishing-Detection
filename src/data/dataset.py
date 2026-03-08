"""Dataset class for loading processed phishing detection data."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import FEATURE_COLUMNS, LABEL_COLUMN

logger = logging.getLogger(__name__)


def load_processed_dataset(
    data_path: Path,
    feature_columns: list[str] | None = None,
    label_column: str = LABEL_COLUMN,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load processed dataset and return feature matrix and labels.

    Args:
        data_path: Path to cleaned CSV (e.g. data/processed/cleaned_dataset.csv).
        feature_columns: Feature column names. Uses FEATURE_COLUMNS if None.
        label_column: Name of label column.

    Returns:
        Tuple of (X: np.ndarray, y: np.ndarray).
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    cols = feature_columns or FEATURE_COLUMNS
    available = [c for c in cols if c in df.columns]

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Columns: {list(df.columns)}")

    X = df[available].values.astype(np.float32)
    y = df[label_column].values.astype(np.int64)

    logger.info("Loaded dataset: %d samples, %d features", X.shape[0], X.shape[1])
    return X, y


def load_with_gan_augmentation(
    data_path: Path,
    gan_phishing_path: Path | None = None,
    gan_phishing_ratio: float = 0.0,
    feature_columns: list[str] | None = None,
    label_column: str = LABEL_COLUMN,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load processed dataset and optionally augment with GAN-generated phishing samples.

    Args:
        data_path: Path to cleaned dataset CSV.
        gan_phishing_path: Path to GAN-generated phishing samples CSV.
        gan_phishing_ratio: Fraction of GAN samples to add (0.0 = none, 0.5 = 50% of phishing count).
        feature_columns: Feature column names.
        label_column: Label column name.
        seed: Random seed for sampling.

    Returns:
        Tuple of (X, y) with optional GAN augmentation.
    """
    X, y = load_processed_dataset(data_path, feature_columns, label_column)

    if gan_phishing_ratio <= 0 or gan_phishing_path is None or not gan_phishing_path.exists():
        return X, y

    rng = np.random.default_rng(seed)
    df_gan = pd.read_csv(gan_phishing_path)

    cols = feature_columns or FEATURE_COLUMNS
    available = [c for c in cols if c in df_gan.columns]
    if label_column in df_gan.columns:
        X_gan = df_gan[available].values.astype(np.float32)
        y_gan = df_gan[label_column].values.astype(np.int64)
    else:
        X_gan = df_gan[available].values.astype(np.float32)
        y_gan = np.ones(len(X_gan), dtype=np.int64)  # GAN phishing = 1

    n_phishing = int(np.sum(y == 1))
    n_gan_to_add = int(n_phishing * gan_phishing_ratio)
    n_gan_to_add = min(n_gan_to_add, len(X_gan))

    if n_gan_to_add > 0:
        indices = rng.choice(len(X_gan), size=n_gan_to_add, replace=False)
        X_gan_sample = X_gan[indices]
        y_gan_sample = y_gan[indices]
        X = np.vstack([X, X_gan_sample])
        y = np.concatenate([y, y_gan_sample])
        logger.info("Augmented with %d GAN phishing samples", n_gan_to_add)

    return X, y


def get_feature_names(feature_columns: list[str] | None = None) -> list[str]:
    """Return list of feature column names."""
    return feature_columns or FEATURE_COLUMNS.copy()
