from typing import Tuple, Optional, Literal, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .nonlinearity_metrics import NonlinearityMetric
from ..model.model import ScaledNet

class EdgeFinder:
    """
    Identifies critical network connections (edges) by aggregating 
    importance metrics over a dataset. Used for both expansion and pruning.
    """
    def __init__(
        self,
        metric: NonlinearityMetric,
        dataloader: DataLoader,
        device: Union[torch.device, str] = torch.device("cpu"),
        aggregation_mode: Literal["mean", "variance"] = "mean",
        max_to_replace: Optional[int] = None,
    ):
        """
        Args:
            metric: The strategy to compute edge importance (e.g., Gradients, Magnitude).
            dataloader: Data source for computing the statistics.
            device: Computation device.
            aggregation_mode: Strategy to combine metrics across batches ('mean' or 'variance').
            max_to_replace: Hard limit on the number of edges to select.
        """
        self.metric = metric
        self.dataloader = dataloader
        self.device = device
        self.aggregation_mode = aggregation_mode
        self.max_to_replace = max_to_replace

    def calculate_edge_metric_for_dataloader(
        self, 
        model: nn.Module, 
        layer: ScaledNet, 
        to_normalise: bool = True, 
        embed: bool = False
    ) -> torch.Tensor:
        """
        Computes the aggregated importance score for every weight in the layer 
        across the entire dataloader.

        Args:
            to_normalise: If True, scales the final metric to the [0, 1] range.
            embed: If True, targets the expansion sub-layer; otherwise, the main weights.
        
        Returns:
            Tensor: A tensor of the same shape as the layer weights containing importance scores.
        """
        if self.aggregation_mode == "mean":
            accumulated: Optional[torch.Tensor] = None
            
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                batch_metric = self.metric.calculate(model, layer, data, target, embed)
                
                if accumulated is None:
                    accumulated = torch.zeros_like(batch_metric).to(self.device)
                accumulated += batch_metric
            
            if accumulated is None:
                raise ValueError("Dataloader is empty.")
                
            aggregated = accumulated / len(self.dataloader)

        elif self.aggregation_mode == "variance":
            sum_: Optional[torch.Tensor] = None
            sum_sq: Optional[torch.Tensor] = None
            n = len(self.dataloader)

            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                batch_metric = self.metric.calculate(model, layer, data, target, embed)

                if sum_ is None or sum_sq is None:
                    sum_ = torch.zeros_like(batch_metric).to(self.device)
                    sum_sq = torch.zeros_like(batch_metric).to(self.device)
                
                sum_ += batch_metric
                sum_sq += batch_metric.pow(2)

            if sum_ is None or sum_sq is None:
                raise ValueError("Dataloader is empty.")

            mean = sum_ / n
            aggregated = (sum_sq / n) - mean.pow(2)
        
        else:
            raise ValueError(f"Unsupported aggregation mode: {self.aggregation_mode}")

        # min max scaling
        if not to_normalise or aggregated.numel() == 0:
            return aggregated

        min_val = aggregated.min()
        max_val = aggregated.max()
        # Add epsilon to avoid division by zero
        normalized = (aggregated - min_val) / (max_val - min_val + 1e-8)
        
        return normalized

    def choose_edges_threshold(
        self, 
        model: nn.Module, 
        layer: ScaledNet, 
        threshold: float, 
        embed: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects edges where the importance metric exceeds the threshold.

        Args:
            threshold: Cut-off value (0.0 to 1.0).
            embed: Whether to check expansion layers or main weights.

        Returns:
            Tuple[Tensor, Tensor]: (parents, children) indices. 
            'Parents' correspond to input features, 'children' to output neurons.
        """
        assert 0 < threshold <= 1, "Threshold must be between 0 and 1."

        # 1. Compute metrics over the dataset
        avg_metric = self.calculate_edge_metric_for_dataloader(
            model=model, layer=layer, to_normalise=True, embed=embed
        )

        # 2. Filter by threshold
        mask = avg_metric > threshold
        
        # 3. Extract indices
        # Weights are (Out, In), so dim0=child (output), dim1=parent (input)
        children, parents = mask.nonzero(as_tuple=True)

        # 4. Limit the number of selected edges (if max_to_replace is set)
        if self.max_to_replace is not None and len(children) > self.max_to_replace:
            masked_metrics = avg_metric[mask]
            # Select top-k highest metrics to prioritize most important edges
            top_k_indices = torch.argsort(masked_metrics, descending=True)[:self.max_to_replace]
            
            children = children[top_k_indices]
            parents = parents[top_k_indices]

        return parents, children
