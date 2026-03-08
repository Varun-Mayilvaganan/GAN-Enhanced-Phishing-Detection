#!/usr/bin/env python3
"""Entry point for training the phishing detection classifier.

Example:
    python train.py --model xgboost --learning_rate 0.01 --epochs 50 --batch_size 64
"""

import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.train_classifier import main

if __name__ == "__main__":
    main()
