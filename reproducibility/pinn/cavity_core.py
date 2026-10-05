"""Neural representation and diagnostic operators for checkpoint evaluation."""

from __future__ import annotations
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self,
        layers: Sequence[int],
        bounds: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
    ) -> None:
        super().__init__()
        if len(layers) < 3:
            raise ValueError("Need at least input, one hidden, and output layers.")
        (self.x_min, self.x_max), (self.y_min, self.y_max), (self.t_min, self.t_max) = bounds
        mods: List[nn.Module] = []
        for i in range(len(layers) - 2):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            mods.append(nn.Tanh())
        mods.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*mods)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("tanh"))
                nn.init.zeros_(m.bias)

    def _normalize(self, xyz: torch.Tensor) -> torch.Tensor:
        x = xyz[:, 0:1]
        y = xyz[:, 1:2]
        t = xyz[:, 2:3]
        x_hat = 2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        y_hat = 2.0 * (y - self.y_min) / (self.y_max - self.y_min) - 1.0
        t_hat = 2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        return torch.cat([x_hat, y_hat, t_hat], dim=1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.net(self._normalize(xyz))
