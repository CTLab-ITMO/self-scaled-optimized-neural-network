from __future__ import annotations

import math
from typing import Optional

import torch.optim as optim


class CosineFullCycleLR:
    """
    За total_epochs эпох плавно поднимает lr от 0 до max_lr (в середине) и опускает до 0 по косинусу.
    lr(epoch) = max_lr * (1 - cos(2π * epoch / total_epochs)) / 2
    """

    def __init__(self, optimizer: optim.Optimizer, max_lr: float, total_epochs: int):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.last_epoch = -1

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        t = epoch / self.total_epochs
        lr = self.max_lr * (1 - math.cos(2 * math.pi * t)) / 2

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
