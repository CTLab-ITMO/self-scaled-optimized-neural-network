import torch

from .nonlinearity_metrics import NonlinearityMetric
from ..model.utils import get_model_last_layer


class EdgeFinder:
    """
    Finds and selects edges in a neural network based on a calculated metric.

    This class provides methods to calculate an edge metric for a given layer of a model,
    and then select edges based on a threshold applied to this metric. It's designed
    to help identify important connections within the network."""
    
    def __init__(
        self,
        metric: NonlinearityMetric,
        dataloader,
        device=torch.device("cpu"),
        aggregation_mode="mean",
        max_to_replace=None,
    ):
        """
        Initializes the NonlinearityAnalyzer with metric, dataloader and device settings.

            Args:
                metric: The nonlinearity metric to use for analysis.
                dataloader: Data loader providing batches of data.
                device:  The PyTorch device to run computations on. Defaults to CPU.
                aggregation_mode: How to aggregate the metric across the dataset ("mean" or "variance").
                                  Defaults to "mean".
                max_to_replace: Maximum number of samples to replace with the calculated metric.

            Returns:
                None
        """
        
        self.max_to_replace = max_to_replace
        self.metric = metric
        self.dataloader = dataloader
        self.device = device
        self.aggregation_mode = aggregation_mode
        assert aggregation_mode in [
            "mean",
            "variance",
        ], "Aggregation mode must be 'mean' or 'variance'."

    def calculate_edge_metric_for_dataloader(
        self, model, layer, to_normalise=True, embed=False
    ):
        """
        Calculates an edge metric for a dataloader using specified aggregation.

            Args:
                model: The model being evaluated.
                layer: The layer for which the metric is calculated.
                to_normalise: Whether to normalise the aggregated metric.
                embed:  Whether to embed the data during metric calculation.

            Returns:
                torch.Tensor: The aggregated and optionally normalised edge metric.
        """
        
        if self.aggregation_mode == "mean":
            accumulated = None
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                metric = self.metric.calculate(model, layer, data, target, embed)
                if accumulated is None:
                    accumulated = torch.zeros_like(metric).to(self.device)
                accumulated += metric
            aggregated = accumulated / len(self.dataloader)
        elif self.aggregation_mode == "variance":
            sum_ = None
            sum_sq = None
            n = len(self.dataloader)
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                metric = self.metric.calculate(model, layer, data, target, embed)
                if sum_ is None:
                    sum_ = torch.zeros_like(metric).to(self.device)
                    sum_sq = torch.zeros_like(metric).to(self.device)
                sum_ += metric
                sum_sq += metric.pow(2)
            mean = sum_ / n
            aggregated = (sum_sq / n) - mean.pow(2)
        else:
            raise ValueError(f"Unsupported aggregation mode: {self.aggregation_mode}")

        if not to_normalise or aggregated.numel() == 0:
            return aggregated

        min_val = aggregated.min()
        max_val = aggregated.max()
        normalized = (aggregated - min_val) / (max_val - min_val + 1e-8)
        return normalized

    # def choose_edges_top_k(self, model, top_k: int, len_choose: int = None):
    #     assert top_k > 0
    #     avg_metric = self.calculate_edge_metric_for_dataloader(model, len_choose)
    #     sorted_indices = torch.argsort(avg_metric, descending=True)
    #     last_layer = get_model_last_layer(model)
    #     return last_layer.weight_indices[:, sorted_indices[:top_k]]

    # def choose_edges_top_percent(self, model, percent: float, len_choose: int = None):
    #     assert 0 < percent <= 1
    #     avg_metric = self.calculate_edge_metric_for_dataloader(model, len_choose)
    #     k = int(percent * avg_metric.numel())
    #     sorted_indices = torch.argsort(avg_metric, descending=True)
    #     last_layer = get_model_last_layer(model)
    #     return last_layer.weight_indices[:, sorted_indices[:k]]

    def choose_edges_threshold(self, model, layer, threshold, embed=False):
        """
        Chooses edges based on a given threshold of their average metric.

            Args:
                model: The model being analyzed.
                layer: The layer whose edges are being selected.
                threshold: The threshold value for selecting edges (between 0 and 1).
                embed: Whether to embed the results or not.

            Returns:
                torch.Tensor: A tensor containing the indices of the chosen edges.
        """
        
        assert 0 < threshold <= 1
        avg_metric = self.calculate_edge_metric_for_dataloader(
            model=model, layer=layer, to_normalise=True, embed=embed
        )
        mask = avg_metric > threshold
        res = layer.weight_indices[:, mask.nonzero(as_tuple=True)[0]]

        if (
            self.max_to_replace is not None
            and self.max_to_replace > 0
            and len(res[0]) > self.max_to_replace
        ):
            
            sorted_indices = torch.argsort(avg_metric[mask], descending=True)
            res = res[:, sorted_indices[: self.max_to_replace]]

        with open('log.txt', 'a') as file:
            print('-'*50, file=file)
            for i in range(len(res[0])):
                print(f'output={res[0][i]}, input={res[1][i]}', file=file)

        return res
    

#    def choose_edges_threshold(self, model, layer, threshold, embed=False):
#        assert 0 < threshold <= 1
#        avg_metric = self.calculate_edge_metric_for_dataloader(
#            model=model, layer=layer, to_normalise=True, embed=embed
#        )
#
#        # Get all edges that meet the threshold
#        mask = avg_metric > threshold
#        all_edges = layer.weight_indices[:, mask.nonzero(as_tuple=True)[0]]
#        all_metrics = avg_metric[mask]
#
#        input_indices = all_edges[1]
#        mask_low = input_indices < 784
#        mask_high = input_indices >= 784
#
#        edges_low = all_edges[:, mask_low]
#        edges_high = all_edges[:, mask_high]
#        metrics_low = all_metrics[mask_low]
#        metrics_high = all_metrics[mask_high]
#
#        if self.max_to_replace is not None and self.max_to_replace > 0:
#            max_edges = self.max_to_replace
#        else:
#            max_edges = len(all_edges[0])
#
#        n_low = min(len(edges_low[0]), max_edges // 2)
#        n_high = min(len(edges_high[0]), max_edges - n_low)
#
#        if n_low > 0:
#            sorted_indices_low = torch.argsort(metrics_low, descending=True)
#            selected_low = edges_low[:, sorted_indices_low[:n_low]]
#        else:
#            selected_low = torch.empty((2, 0), dtype=all_edges.dtype, device=all_edges.device)
#
#        if n_high > 0:
#            sorted_indices_high = torch.argsort(metrics_high, descending=True)
#            selected_high = edges_high[:, sorted_indices_high[:n_high]]
#        else:
#            selected_high = torch.empty((2, 0), dtype=all_edges.dtype, device=all_edges.device)
#
#        res = torch.cat([selected_low, selected_high], dim=1)
#        
#        return res
#