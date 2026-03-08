"""Train phishing detection classifier with optional GAN augmentation."""

import argparse
import csv
import logging
from pathlib import Path
from uuid import uuid4

from sklearn.model_selection import train_test_split

from src.data.dataset import get_feature_names, load_with_gan_augmentation
from src.evaluation.error_analysis import run_error_analysis
from src.evaluation.metrics import compute_metrics, print_metrics
from src.models.classifier import predict, predict_proba, train_classifier
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default experiment log path
EXPERIMENT_LOG = Path("experiments/experiment_log.csv")


def log_experiment(
    experiment_id: str,
    model: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    f1_score: float,
    precision: float,
    recall: float,
    roc_auc: float,
    accuracy: float,
    log_path: Path = EXPERIMENT_LOG,
) -> None:
    """Append experiment results to CSV log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "experiment_id", "model", "learning_rate", "batch_size", "epochs",
                "F1_score", "precision", "recall", "roc_auc", "accuracy",
            ])
        writer.writerow([
            experiment_id, model, learning_rate, batch_size, epochs,
            f1_score, precision, recall, roc_auc, accuracy,
        ])
    logger.info("Logged experiment to %s", log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phishing detection classifier")
    parser.add_argument(
        "--model",
        type=str,
        choices=["xgboost", "lightgbm", "catboost"],
        default="xgboost",
        help="Classifier model",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/cleaned_dataset.csv"),
        help="Path to cleaned dataset",
    )
    parser.add_argument(
        "--gan-samples",
        type=Path,
        default=None,
        help="Path to GAN phishing samples for augmentation",
    )
    parser.add_argument(
        "--gan-ratio",
        type=float,
        default=0.0,
        help="Fraction of GAN samples to add (0 = none, 0.5 = 50%% of phishing count)",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of boosting rounds (n_estimators)")
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=0.01, dest="learning_rate", help="Learning rate")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=64, dest="batch_size", help="Batch size (for logging)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--experiment-log",
        type=Path,
        default=EXPERIMENT_LOG,
        help="Path to experiment log CSV",
    )
    parser.add_argument(
        "--error-analysis",
        type=Path,
        default=None,
        help="Path to save error analysis results",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    if not args.data_path.exists():
        logger.error("Dataset not found: %s. Run preprocess first.", args.data_path)
        return

    X, y = load_with_gan_augmentation(
        data_path=args.data_path,
        gan_phishing_path=args.gan_samples,
        gan_phishing_ratio=args.gan_ratio,
        seed=args.seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    model = train_classifier(
        X_train=X_train,
        y_train=y_train,
        model_type=args.model,
        learning_rate=args.learning_rate,
        n_estimators=args.epochs,
        random_state=args.seed,
    )

    y_pred = predict(model, X_test)
    y_prob = predict_proba(model, X_test)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    print_metrics(metrics)

    experiment_id = str(uuid4())[:8]
    log_experiment(
        experiment_id=experiment_id,
        model=args.model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        f1_score=metrics["f1_score"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        roc_auc=metrics["roc_auc"],
        accuracy=metrics["accuracy"],
        log_path=args.experiment_log,
    )

    if args.error_analysis:
        feature_names = get_feature_names()
        run_error_analysis(
            X_test, y_test, y_pred,
            feature_names=feature_names,
            output_path=args.error_analysis,
        )


if __name__ == "__main__":
    main()
