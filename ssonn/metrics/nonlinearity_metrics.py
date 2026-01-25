from abc import abstractmethod, ABC
from typing import Callable, Optional

import torch
import torch.nn as nn

from ssonn.model.model import ScaledNet


class NonlinearityMetric(ABC):
    """
    Base abstract class for computing connection importance metrics.
    Used to determine which edges to expand.
    """
    # TODO: move model to init
    def __init__(self, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """
        Args:
            loss_fn: Loss function used for gradient-based calculations.
        """
        self.loss_fn = loss_fn

    @abstractmethod
    def calculate(
        self, 
        model: nn.Module, 
        layer: ScaledNet, 
        X_arr: Optional[torch.Tensor], 
        y_arr: Optional[torch.Tensor], 
        embed: bool = False
    ) -> torch.Tensor:
        """
        Computes the importance metric for the layer's weights.

        Args:
            model: The full neural network.
            layer: The ScaledNet layer being analyzed.
            X_arr: Input batch for gradient computation (optional for magnitude metrics).
            y_arr: Target batch for gradient computation (optional for magnitude metrics).
            embed: If True, calculates metric for the last expansion sub-layer.
                   If False, calculates for the main projection weights.

        Returns:
            torch.Tensor: Tensor of the same shape as the target weights containing importance scores.
        """
        pass


class AbsGradientEdgeMetric(NonlinearityMetric):
    """
    Estimates importance based on the absolute value of gradients.
    High gradient implies the loss is sensitive to this connection.
    """
    def calculate(
        self, 
        model: nn.Module, 
        layer: ScaledNet, 
        X_arr: Optional[torch.Tensor], 
        y_arr: Optional[torch.Tensor], 
        embed: bool = False
    ) -> torch.Tensor:
        if X_arr is None or y_arr is None:
            raise ValueError("X_arr and y_arr are required for gradient computation")

        model.eval()
        model.zero_grad()

        y_pred = model(X_arr).squeeze()
        loss = self.loss_fn(y_pred, y_arr)
        loss.backward()
        
        if embed:
            if len(layer.sub_linears) > 0:
                target_param = layer.sub_linears[-1].weight
            else:
                return torch.zeros_like(layer.weight)
        else:
            target_param = layer.weight
            
        if target_param.grad is None:
            raise ValueError("Impossible to calculate the metric. Most likely the layer is frozen")

        edge_gradients = torch.abs(target_param.grad)
        model.zero_grad()
        return edge_gradients   


class MagnitudeL1Metric(NonlinearityMetric):
    """
    Estimates importance based on weight magnitude (L1 Norm).
    """
    def calculate(
        self, 
        model: nn.Module, 
        layer: ScaledNet, 
        X_arr: Optional[torch.Tensor], 
        y_arr: Optional[torch.Tensor], 
        embed: bool = False
    ) -> torch.Tensor:
        if embed:
            if len(layer.sub_linears) > 0:
                return torch.abs(layer.sub_linears[-1].weight)
            return torch.zeros_like(layer.weight)
        return layer.weight.abs()


class MagnitudeL2Metric(NonlinearityMetric):
    """
    Estimates importance based on weight magnitude (L2 Norm).
    """
    def calculate(
        self, 
        model: nn.Module, 
        layer: ScaledNet, 
        X_arr: Optional[torch.Tensor], 
        y_arr: Optional[torch.Tensor], 
        embed: bool = False
    ) -> torch.Tensor:
        if embed:
            if len(layer.sub_linears) > 0:
                return torch.pow(layer.sub_linears[-1].weight, 2)
            return torch.zeros_like(layer.weight)
        return torch.pow(layer.weight, 2)
