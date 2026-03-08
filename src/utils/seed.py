"""Reproducibility utilities - seed control for Python, NumPy, and PyTorch."""

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Random seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        logger.info("Seeds set: Python, NumPy, PyTorch (seed=%d)", seed)
    except ImportError:
        logger.info("Seeds set: Python, NumPy (seed=%d). PyTorch not installed.", seed)
