"""Unit tests for dataset loading and validation."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import load_processed_dataset, load_with_gan_augmentation
from src.data.preprocess import FEATURE_COLUMNS, LABEL_COLUMN


@pytest.fixture
def sample_dataset_path() -> Path:
    """Create a temporary valid dataset CSV."""
    df = pd.DataFrame(
        np.random.randint(0, 2, size=(100, 16)),
        columns=FEATURE_COLUMNS + [LABEL_COLUMN],
    )
    df[LABEL_COLUMN] = np.random.randint(0, 2, size=100)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = Path(f.name)
    df.to_csv(path, index=False)
    yield path
    path.unlink(missing_ok=True)


def test_dataset_loads_correctly(sample_dataset_path: Path) -> None:
    """Verify dataset loads without error and returns correct types."""
    X, y = load_processed_dataset(sample_dataset_path)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.dtype in (np.float32, np.float64)
    assert y.dtype in (np.int32, np.int64)


def test_label_column_exists(sample_dataset_path: Path) -> None:
    """Verify label column is present and extracted."""
    X, y = load_processed_dataset(sample_dataset_path)
    assert len(y) == len(X)
    assert set(np.unique(y)).issubset({0, 1})


def test_dataset_shape_is_valid(sample_dataset_path: Path) -> None:
    """Verify dataset has valid shape (n_samples, n_features)."""
    X, y = load_processed_dataset(sample_dataset_path)
    assert X.ndim == 2
    assert X.shape[0] > 0
    assert X.shape[1] == len(FEATURE_COLUMNS)
    assert y.shape[0] == X.shape[0]


def test_load_nonexistent_file_raises() -> None:
    """Verify FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_processed_dataset(Path("/nonexistent/path/data.csv"))


def test_load_without_label_column_raises() -> None:
    """Verify ValueError when label column is missing."""
    df = pd.DataFrame(np.random.rand(10, 5), columns=["a", "b", "c", "d", "e"])
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = Path(f.name)
    df.to_csv(path, index=False)
    try:
        with pytest.raises(ValueError, match="Label column"):
            load_processed_dataset(path)
    finally:
        path.unlink(missing_ok=True)


def test_load_with_gan_augmentation_no_gan(sample_dataset_path: Path) -> None:
    """Verify load_with_gan_augmentation returns same as load when gan_ratio=0."""
    X1, y1 = load_processed_dataset(sample_dataset_path)
    X2, y2 = load_with_gan_augmentation(sample_dataset_path, gan_phishing_ratio=0.0)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
