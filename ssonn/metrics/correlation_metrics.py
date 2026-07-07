"""
Correlation metrics for analyzing the relationship between neuron activation
and outgoing edge weights in SSONN.

Measures:
- Binary activation: fraction of samples where neuron's pre-activation > 0
- ReLU value activation: mean absolute LeakyReLU output value
- Edge weights: mean |w| of outgoing edges (selected/unselected/all)
- Pearson and Spearman correlations between activation and edge weights
"""

from typing import Dict, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from ..model.model import ScaledNet


def compute_activation_stats(
    model: nn.Module,
    target_layer: ScaledNet,
    dataloader: DataLoader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes activation statistics for all input features of the target ScaledNet layer.

    For original input features (pixels):
      - binary: fraction of samples where input > 0
      - value: mean absolute input value

    For SubLinear neurons:
      - binary: fraction of samples where pre-activation > 0 (active regime of LeakyReLU)
      - value: mean absolute post-activation (LeakyReLU) value

    Returns:
        (binary_activation, activation_value): tensors of shape (in_features,)
    """
    model.eval()

    in_features = target_layer.in_features
    sub_dims = [sub.weight.shape[0] for sub in target_layer.sub_linears]
    original_dim = in_features - sum(sub_dims)

    sum_binary = torch.zeros(in_features, device=device)
    sum_value = torch.zeros(in_features, device=device)
    n_samples = 0

    # Register hooks on each SubLinear's LeakyReLU activation
    hooks = []
    offset = original_dim

    for i, sub in enumerate(target_layer.sub_linears):
        dim_i = sub_dims[i]

        def make_hook(off, dim):
            def hook_fn(module, inp, out):
                # inp[0] = pre-activation (input to LeakyReLU)
                # out = post-activation (output of LeakyReLU)
                pre_act = inp[0]
                post_act = out
                sum_binary[off:off + dim] += (pre_act > 0).float().sum(dim=0)
                sum_value[off:off + dim] += post_act.abs().sum(dim=0)
            return hook_fn

        hooks.append(sub.act.register_forward_hook(make_hook(offset, dim_i)))
        offset += dim_i

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            batch_size = data.shape[0]
            n_samples += batch_size

            # Original input features
            sum_binary[:original_dim] += (data > 0).float().sum(dim=0)
            sum_value[:original_dim] += data.abs().sum(dim=0)

            # Forward pass triggers SubLinear hooks
            _ = model(data)

    for h in hooks:
        h.remove()

    if n_samples == 0:
        return torch.zeros(in_features), torch.zeros(in_features)

    binary_act = (sum_binary / n_samples).cpu()
    activation_val = (sum_value / n_samples).cpu()

    return binary_act, activation_val


def compute_edge_weight_stats(
    target_layer: ScaledNet,
    selected_parents: torch.Tensor,
    selected_children: torch.Tensor,
) -> Dict[str, Any]:
    """
    Computes per-feature outgoing edge weight statistics from ScaledNet.weight.

    Selected edges = chosen by the expansion metric for this expansion.
    Unselected edges = non-zero edges NOT chosen by the metric.

    Args:
        target_layer: The ScaledNet layer.
        selected_parents: Column indices (input features) of edges selected for expansion.
        selected_children: Row indices (output neurons) of edges selected for expansion.

    Returns:
        Dict with per-feature tensors and aggregate scalars.
    """
    weight = target_layer.weight.data.detach().cpu()
    abs_weight = weight.abs()

    # Build masks
    selected_mask = torch.zeros_like(weight, dtype=torch.bool)
    if len(selected_parents) > 0:
        sp = selected_parents.cpu().long()
        sc = selected_children.cpu().long()
        selected_mask[sc, sp] = True

    nonzero_mask = abs_weight > 1e-9
    unselected_mask = nonzero_mask & ~selected_mask

    # Per-column (per-feature) statistics — vectorized
    def per_col_stats(mask):
        counts = mask.float().sum(dim=0)
        sums = (abs_weight * mask.float()).sum(dim=0)
        means = torch.where(counts > 0, sums / counts, torch.zeros_like(counts))
        return means, counts

    mean_selected, n_selected = per_col_stats(selected_mask)
    mean_unselected, n_unselected = per_col_stats(unselected_mask)
    mean_all, n_all = per_col_stats(nonzero_mask)

    # Aggregate (next-layer) statistics
    def agg_mean(mask):
        vals = abs_weight[mask]
        return float(vals.mean().item()) if len(vals) > 0 else 0.0

    return {
        'mean_abs_selected': mean_selected,
        'mean_abs_unselected': mean_unselected,
        'mean_abs_all': mean_all,
        'n_selected': n_selected,
        'n_unselected': n_unselected,
        'n_all': n_all,
        'next_layer_mean_selected': agg_mean(selected_mask),
        'next_layer_mean_unselected': agg_mean(unselected_mask),
        'next_layer_mean_all': agg_mean(nonzero_mask),
        'total_selected': int(selected_mask.sum().item()),
        'total_unselected': int(unselected_mask.sum().item()),
        'total_all': int(nonzero_mask.sum().item()),
        'fraction_selected': float(selected_mask.sum().item()) / max(int(nonzero_mask.sum().item()), 1),
    }


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Computes Pearson and Spearman correlations with safety checks.
    Returns (pearson_r, spearman_r). NaN if insufficient data or zero variance.
    """
    if len(x) < 3 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return float('nan'), float('nan')

    pear_r, _ = pearsonr(x, y)
    spear_r, _ = spearmanr(x, y)
    return float(pear_r), float(spear_r)


def compute_correlations(
    binary_act: torch.Tensor,
    relu_act: torch.Tensor,
    edge_stats: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Computes Pearson and Spearman correlations between activation and edge weight stats.
    """
    results = {}

    for act_name, act_data in [('binary', binary_act), ('relu_value', relu_act)]:
        for edge_name in ['selected', 'unselected', 'all']:
            edge_data = edge_stats[f'mean_abs_{edge_name}']
            n_edges = edge_stats[f'n_{edge_name}']

            # Only include features with at least one edge of this type
            mask = n_edges > 0

            if mask.sum() < 3:
                results[f'corr/pearson/{act_name}_vs_{edge_name}_edges'] = float('nan')
                results[f'corr/spearman/{act_name}_vs_{edge_name}_edges'] = float('nan')
                continue

            x = act_data[mask].numpy().astype(np.float64)
            y = edge_data[mask].numpy().astype(np.float64)

            pear_r, spear_r = _safe_correlation(x, y)
            results[f'corr/pearson/{act_name}_vs_{edge_name}_edges'] = pear_r
            results[f'corr/spearman/{act_name}_vs_{edge_name}_edges'] = spear_r

    return results


def create_scatter_plots(
    binary_act: torch.Tensor,
    relu_act: torch.Tensor,
    edge_stats: Dict[str, torch.Tensor],
    epoch: int,
) -> Dict[str, plt.Figure]:
    """
    Creates scatter plots of activation vs edge weights for each edge type.
    """
    figures = {}

    for edge_name in ['selected', 'unselected', 'all']:
        edge_data = edge_stats[f'mean_abs_{edge_name}']
        n_edges = edge_stats[f'n_{edge_name}']
        mask = n_edges > 0

        if mask.sum() < 2:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        y = edge_data[mask].numpy()

        # Binary activation vs edges
        x_bin = binary_act[mask].numpy()
        axes[0].scatter(x_bin, y, alpha=0.3, s=10, c='steelblue')
        axes[0].set_xlabel('Binary Activation (fraction > 0)')
        axes[0].set_ylabel(f'Mean |weight| ({edge_name} edges)')
        axes[0].set_title(f'Binary Act vs {edge_name.capitalize()} Edges (Epoch {epoch})')
        axes[0].grid(True, alpha=0.3)

        # ReLU value vs edges
        x_relu = relu_act[mask].numpy()
        axes[1].scatter(x_relu, y, alpha=0.3, s=10, c='coral')
        axes[1].set_xlabel('Mean |Activation Value| (LeakyReLU)')
        axes[1].set_ylabel(f'Mean |weight| ({edge_name} edges)')
        axes[1].set_title(f'ReLU Value vs {edge_name.capitalize()} Edges (Epoch {epoch})')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        figures[f'scatter/{edge_name}_edges'] = fig

    return figures


def log_correlation_metrics_at_expansion(
    model: nn.Module,
    target_layer: ScaledNet,
    dataloader: DataLoader,
    device: str,
    selected_parents: torch.Tensor,
    selected_children: torch.Tensor,
    experiment: Any,
    epoch: int,
) -> None:
    """
    Computes and logs all correlation metrics at expansion time.

    Logs Pearson & Spearman correlations for binary/relu × selected/unselected/all edges,
    aggregate stats, scatter plots, and per-SubLinear correlations.
    """
    print(f"  [Correlation] Computing activation-edge correlations at expansion (epoch {epoch})...")

    # 1. Compute activation stats
    binary_act, relu_act = compute_activation_stats(model, target_layer, dataloader, device)

    # 2. Compute edge weight stats
    edge_stats = compute_edge_weight_stats(target_layer, selected_parents, selected_children)

    # 3. Compute correlations
    correlations = compute_correlations(binary_act, relu_act, edge_stats)

    # 4. Build log dict
    logs = {}
    logs.update(correlations)

    # Aggregate stats
    logs['activation/mean_binary_activation'] = float(binary_act.mean().item())
    logs['activation/mean_relu_value'] = float(relu_act.mean().item())
    logs['edges/mean_abs_weight_selected'] = edge_stats['next_layer_mean_selected']
    logs['edges/mean_abs_weight_unselected'] = edge_stats['next_layer_mean_unselected']
    logs['edges/mean_abs_weight_all'] = edge_stats['next_layer_mean_all']
    logs['edges/total_selected'] = edge_stats['total_selected']
    logs['edges/total_unselected'] = edge_stats['total_unselected']
    logs['edges/fraction_selected'] = edge_stats['fraction_selected']

    experiment.log_metrics(logs, step=epoch)

    # 5. Scatter plots
    figures = create_scatter_plots(binary_act, relu_act, edge_stats, epoch)
    for name, fig in figures.items():
        experiment.log_figure(name, fig, step=epoch)
        plt.close(fig)

    # 6. Per-SubLinear correlations
    sub_dims = [sub.weight.shape[0] for sub in target_layer.sub_linears]
    if len(sub_dims) > 0:
        original_dim = target_layer.in_features - sum(sub_dims)
        offset = original_dim

        for sub_idx, dim_k in enumerate(sub_dims):
            sl = slice(offset, offset + dim_k)
            sub_binary = binary_act[sl]
            sub_relu = relu_act[sl]

            sub_edge_stats = {}
            for t in ['selected', 'unselected', 'all']:
                sub_edge_stats[f'mean_abs_{t}'] = edge_stats[f'mean_abs_{t}'][sl]
                sub_edge_stats[f'n_{t}'] = edge_stats[f'n_{t}'][sl]

            sub_corr = compute_correlations(sub_binary, sub_relu, sub_edge_stats)
            sub_logs = {
                key.replace('corr/', f'corr/sub{sub_idx}/'): v
                for key, v in sub_corr.items()
            }
            experiment.log_metrics(sub_logs, step=epoch)

            offset += dim_k

    print(f"  [Correlation] Logged {len(logs)} metrics + {len(figures)} scatter plots.")


def log_correlation_metrics_epoch(
    model: nn.Module,
    target_layer: ScaledNet,
    dataloader: DataLoader,
    device: str,
    experiment: Any,
    epoch: int,
) -> None:
    """
    Computes and logs all-edges correlation metrics every epoch.
    Only logs Pearson/Spearman for binary/relu × all edges + aggregate stats.
    """
    # 1. Compute activation stats
    binary_act, relu_act = compute_activation_stats(model, target_layer, dataloader, device)

    # 2. Edge stats (all non-zero edges, no selected/unselected distinction)
    weight = target_layer.weight.data.detach().cpu()
    abs_weight = weight.abs()
    nonzero_mask = abs_weight > 1e-9

    n_all = nonzero_mask.float().sum(dim=0)
    sum_all = (abs_weight * nonzero_mask.float()).sum(dim=0)
    mean_all = torch.where(n_all > 0, sum_all / n_all, torch.zeros_like(n_all))

    # 3. Correlations (all-edges only)
    logs = {}

    for act_name, act_data in [('binary', binary_act), ('relu_value', relu_act)]:
        mask = n_all > 0

        if mask.sum() >= 3:
            x = act_data[mask].numpy().astype(np.float64)
            y = mean_all[mask].numpy().astype(np.float64)
            pear_r, spear_r = _safe_correlation(x, y)
        else:
            pear_r, spear_r = float('nan'), float('nan')

        logs[f'corr/pearson/{act_name}_vs_all_edges'] = pear_r
        logs[f'corr/spearman/{act_name}_vs_all_edges'] = spear_r

    # 4. Aggregate stats
    logs['activation/mean_binary_activation'] = float(binary_act.mean().item())
    logs['activation/mean_relu_value'] = float(relu_act.mean().item())

    all_nonzero = abs_weight[nonzero_mask]
    logs['edges/mean_abs_weight_all'] = float(all_nonzero.mean().item()) if len(all_nonzero) > 0 else 0.0

    experiment.log_metrics(logs, step=epoch)
