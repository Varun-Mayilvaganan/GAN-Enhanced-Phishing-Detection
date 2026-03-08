"""GAN model for generating synthetic phishing samples."""

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """Generator network: maps latent noise to synthetic feature vectors."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (128, 256),
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    """Discriminator network: classifies real vs generated samples."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        leaky_slope: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LeakyReLU(leaky_slope)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PhishingGAN:
    """
    GAN for generating synthetic phishing feature samples.

    Trains on phishing-class samples and generates new samples
    to augment the training dataset.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        data_dim: int | None = None,
        device: torch.device | None = None,
        seed: int = 42,
    ) -> None:
        set_seed(seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.data_dim = data_dim
        self.generator: Generator | None = None
        self.discriminator: Discriminator | None = None

    def _build_models(self, data_dim: int) -> None:
        self.data_dim = data_dim
        self.generator = Generator(self.latent_dim, data_dim).to(self.device)
        self.discriminator = Discriminator(data_dim).to(self.device)

    def train(
        self,
        data: np.ndarray,
        num_epochs: int = 500,
        batch_size: int = 64,
        lr: float = 0.0002,
        log_every: int = 100,
        early_stop_epoch: int | None = None,
    ) -> None:
        """
        Train the GAN on phishing feature data.

        Args:
            data: 2D array of shape (n_samples, n_features).
            num_epochs: Number of training epochs.
            batch_size: Batch size for training.
            lr: Learning rate for Adam.
            log_every: Log loss every N epochs.
            early_stop_epoch: Stop training at this epoch if set.
        """
        data_dim = data.shape[1]
        self._build_models(data_dim)

        dataset = TensorDataset(torch.tensor(data, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        criterion = nn.BCELoss()

        for epoch in range(num_epochs):
            for (real_batch,) in dataloader:
                real_batch = real_batch.to(self.device)
                batch_size_actual = real_batch.size(0)

                real_labels = torch.ones(batch_size_actual, 1, device=self.device)
                fake_labels = torch.zeros(batch_size_actual, 1, device=self.device)

                # Train Discriminator
                optimizer_D.zero_grad()
                d_real = self.discriminator(real_batch)
                d_loss_real = criterion(d_real, real_labels)

                z = torch.randn(batch_size_actual, self.latent_dim, device=self.device)
                fake_batch = self.generator(z)
                d_fake = self.discriminator(fake_batch.detach())
                d_loss_fake = criterion(d_fake, fake_labels)

                d_loss = (d_loss_real + d_loss_fake) / 2
                d_loss.backward()
                optimizer_D.step()

                # Train Generator
                optimizer_G.zero_grad()
                z = torch.randn(batch_size_actual, self.latent_dim, device=self.device)
                gen_samples = self.generator(z)
                g_output = self.discriminator(gen_samples)
                g_loss = criterion(g_output, real_labels)
                g_loss.backward()
                optimizer_G.step()

            if (epoch + 1) % log_every == 0:
                logger.info(
                    "Epoch %d | D_loss: %.4f | G_loss: %.4f",
                    epoch + 1,
                    d_loss.item(),
                    g_loss.item(),
                )

            if early_stop_epoch is not None and epoch + 1 >= early_stop_epoch:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    def generate(
        self,
        n_samples: int,
        binarize: bool = True,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Generate synthetic phishing samples.

        Args:
            n_samples: Number of samples to generate.
            binarize: If True, binarize outputs (>threshold -> 1).
            threshold: Binarization threshold.

        Returns:
            Generated samples as numpy array of shape (n_samples, n_features).
        """
        if self.generator is None:
            raise RuntimeError("GAN not trained. Call train() first.")

        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(z).cpu().numpy()

        if binarize:
            samples = (samples > threshold).astype(np.float32)

        return samples
