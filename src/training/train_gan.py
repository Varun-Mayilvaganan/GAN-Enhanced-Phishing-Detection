"""Train GAN to generate synthetic phishing samples."""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data.dataset import load_processed_dataset
from src.models.gan import PhishingGAN
from src.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GAN for synthetic phishing data")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/cleaned_dataset.csv"),
        help="Path to cleaned dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory to save generated samples",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent dimension")
    parser.add_argument("--n-samples", type=int, default=2000, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    if not args.data_path.exists():
        logger.error("Dataset not found: %s. Run preprocess first.", args.data_path)
        return

    X, y = load_processed_dataset(args.data_path)
    phishing_mask = y == 1
    X_phishing = X[phishing_mask]

    if len(X_phishing) == 0:
        logger.error("No phishing samples in dataset.")
        return

    logger.info("Training GAN on %d phishing samples", len(X_phishing))

    gan = PhishingGAN(latent_dim=args.latent_dim, seed=args.seed)
    gan.train(
        data=X_phishing,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        log_every=100,
        early_stop_epoch=args.epochs,
    )

    samples = gan.generate(n_samples=args.n_samples, binarize=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_names = [
        "Have_IP", "Have_At", "URL_Length", "URL_Depth", "Redirection",
        "https_Domain", "TinyURL", "Prefix/Suffix", "DNS_Record", "Domain_Age",
        "Domain_End", "iFrame", "Mouse_Over", "Right_Click", "Web_Forwards",
    ]
    df = pd.DataFrame(samples, columns=feature_names)
    df["Label"] = 1

    output_file = args.output_dir / "GAN_phishing_samples.csv"
    df.to_csv(output_file, index=False)
    logger.info("Saved %d GAN samples to %s", len(df), output_file)


if __name__ == "__main__":
    main()
