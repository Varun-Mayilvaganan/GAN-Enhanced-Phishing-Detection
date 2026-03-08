"""Data preprocessing: load, clean, scale, and save phishing detection dataset."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Missing value markers to replace
MISSING_VALUES = ["-", "(empty)", "nan", "NaN", ""]

# Expected feature columns (excluding Domain and Label)
FEATURE_COLUMNS = [
    "Have_IP",
    "Have_At",
    "URL_Length",
    "URL_Depth",
    "Redirection",
    "https_Domain",
    "TinyURL",
    "Prefix/Suffix",
    "DNS_Record",
    "Domain_Age",
    "Domain_End",
    "iFrame",
    "Mouse_Over",
    "Right_Click",
    "Web_Forwards",
]

LABEL_COLUMN = "Label"


def load_dataset(
    data_dir: Path,
    legitimate_file: str = "legitimate.csv",
    phishing_file: str = "phishing.csv",
) -> pd.DataFrame:
    """
    Load legitimate and phishing datasets from directory and combine them.

    Args:
        data_dir: Directory containing CSV files.
        legitimate_file: Filename for legitimate samples.
        phishing_file: Filename for phishing samples.

    Returns:
        Combined DataFrame with both classes.

    Raises:
        FileNotFoundError: If required files are missing.
    """
    legit_path = data_dir / legitimate_file
    phish_path = data_dir / phishing_file

    if not legit_path.exists():
        raise FileNotFoundError(f"Legitimate data not found: {legit_path}")
    if not phish_path.exists():
        raise FileNotFoundError(f"Phishing data not found: {phish_path}")

    df_legit = pd.read_csv(legit_path)
    df_phish = pd.read_csv(phish_path)

    df = pd.concat([df_legit, df_phish], ignore_index=True)
    logger.info("Loaded %d legitimate + %d phishing = %d total samples", len(df_legit), len(df_phish), len(df))
    return df


def replace_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace common missing value markers with NaN.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with missing values as NaN.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype.name == "string":
            df[col] = df[col].replace(MISSING_VALUES, np.nan)
        else:
            df[col] = df[col].replace(MISSING_VALUES, np.nan)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Deduplicated DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before > after:
        logger.info("Removed %d duplicate rows", before - after)
    return df


def prepare_features_and_labels(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    label_column: str = LABEL_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix and labels, dropping non-feature columns.

    Args:
        df: Input DataFrame.
        feature_columns: List of feature column names. Uses FEATURE_COLUMNS if None.
        label_column: Name of label column.

    Returns:
        Tuple of (features DataFrame, labels Series).
    """
    cols = feature_columns or FEATURE_COLUMNS
    available = [c for c in cols if c in df.columns]
    missing = set(cols) - set(available)
    if missing:
        logger.warning("Feature columns not found (will be dropped): %s", missing)

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Columns: {list(df.columns)}")

    X = df[available].copy()
    y = df[label_column].copy()

    # Fill remaining NaN with 0 for numeric features (binary/multi-category)
    X = X.fillna(0)

    # Ensure numeric types
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(np.float32)

    y = pd.to_numeric(y, errors="coerce")
    invalid_labels = y.isna()
    if invalid_labels.any():
        logger.warning("Dropping %d rows with invalid labels", invalid_labels.sum())
        X = X[~invalid_labels]
        y = y[~invalid_labels]

    y = y.astype(int)
    return X, y


def scale_features(
    X: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[np.ndarray, StandardScaler | None]:
    """
    Apply StandardScaler to feature matrix.

    Args:
        X: Feature DataFrame.
        scaler: Fitted scaler for transform-only. If None and fit=True, creates new.
        fit: If True, fit the scaler; otherwise only transform.

    Returns:
        Tuple of (scaled array, scaler).
    """
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled.astype(np.float32), scaler


def preprocess(
    input_dir: Path,
    output_dir: Path,
    output_file: str = "cleaned_dataset.csv",
    seed: int = 42,
) -> Path:
    """
    Full preprocessing pipeline: load, clean, scale, save.

    Args:
        input_dir: Directory with legitimate.csv and phishing.csv.
        output_dir: Directory to save cleaned dataset.
        output_file: Output filename.
        seed: Random seed for reproducibility.

    Returns:
        Path to saved cleaned dataset.
    """
    set_seed(seed)

    df = load_dataset(input_dir)
    df = replace_missing_values(df)
    df = remove_duplicates(df)

    X, y = prepare_features_and_labels(df)
    X_scaled, scaler = scale_features(X, fit=True)

    result = pd.DataFrame(X_scaled, columns=X.columns)
    result[LABEL_COLUMN] = y.values

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_file
    result.to_csv(output_path, index=False)
    logger.info("Saved cleaned dataset to %s (%d samples, %d features)", output_path, len(result), len(X.columns))

    # Save scaler for inference
    import joblib

    scaler_path = output_dir / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    logger.info("Saved scaler to %s", scaler_path)

    return output_path


def main() -> None:
    """CLI entry point for preprocessing."""
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Preprocess phishing detection dataset")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory with legitimate.csv and phishing.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save cleaned dataset",
    )
    parser.add_argument("--output-file", default="cleaned_dataset.csv", help="Output filename")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    preprocess(args.input_dir, args.output_dir, args.output_file, args.seed)


if __name__ == "__main__":
    main()
