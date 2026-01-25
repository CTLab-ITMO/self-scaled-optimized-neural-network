import copy
from abc import abstractmethod, ABC

import torch
from torch import nn

from ..model.model import ExpandingLinear
from ..model.utils import get_model_last_layer, unfreeze_all


class NonlinearityMetric(ABC):
    # TODO: move model to init
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn

    @abstractmethod
    def calculate(self, model, layer, X_arr, y_arr, embed=False) -> torch.Tensor:
        pass


class AbsGradientEdgeMetric(NonlinearityMetric):
    def calculate(self, model, layer, X_arr, y_arr, embed=False) -> torch.Tensor:
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
            
        edge_gradients = target_param.grad.abs()
        model.zero_grad()
        return edge_gradients   


class MagnitudeL1Metric(NonlinearityMetric):
    def calculate(self, model, layer, X_arr=None, y_arr=None, embed=False) -> torch.Tensor:
        if embed:
            if len(layer.sub_linears) > 0:
                return layer.sub_linears[-1].weight.abs()
            return torch.zeros_like(layer.weight)
        return layer.weight.abs()


class MagnitudeL2Metric(NonlinearityMetric):
    def calculate(self, model, layer, X_arr=None, y_arr=None, embed=False) -> torch.Tensor:
        if embed:
            if len(layer.sub_linears) > 0:
                return torch.pow(layer.sub_linears[-1].weight, 2)
            return torch.zeros_like(layer.weight)
        return torch.pow(layer.weight, 2)
