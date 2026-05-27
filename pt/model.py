"""PyTorch architecture for German credit classification (train + inference)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class NormalizedCreditNet(nn.Module):
    """MLP with StandardScaler-style normalization baked into the graph."""

    def __init__(self, n_in: int, n_out: int, X_train: np.ndarray, hidden: int = 32):
        super().__init__()
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.net(x)


# Backward compatibility for models saved before rename.
NormalizedIrisNet = NormalizedCreditNet
