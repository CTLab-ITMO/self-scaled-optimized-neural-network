from typing import Any, Tuple

import matplotlib.pyplot as plt 
import numpy as np
from torch import nn
import torch

from ..model.model import LowRankAdapter, ScaledNet, SubLinear


def log_gradient_image(model: nn.Module, experiment: Any, epoch: int):
    """
    Visualizes gradients of the first 784 inputs (original pixels) for the main projection layer.
    """
    if not hasattr(model, 'weight') or model.weight.grad is None:
        return

    # Gradients shape: (10, In_Features). We take only the first 784 (pixels).
    # Assuming the first 784 columns correspond to the original image input.
    grad = model.weight.grad.detach().cpu()

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(10): # For each class
        if i >= grad.shape[0]: break
        
        # Reshape the class row vector to 28x28
        grad_img = grad[i, :784].view(28, 28).numpy()
        
        im = axes[i].imshow(grad_img, cmap='viridis')
        axes[i].set_title(f'Class {i} Grads')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    experiment.log_figure(f"gradients_image", fig, step=epoch)
    plt.close(fig)


def log_gradient_values(name: str, grad_tensor: torch.Tensor, experiment: Any, epoch: int):
    """Plots the raw values of gradients sorted by index."""
    if grad_tensor is None: return
    
    # Flatten and take mean or just flatten? 
    # Original code did: torch.stack(gradients).mean(dim=0). 
    # Here we have a snapshot of gradients. Let's flatten.
    flat_grad = grad_tensor.detach().cpu().flatten().numpy()
    
    # Downsample for plotting if too large
    if len(flat_grad) > 10000:
        indices = np.linspace(0, len(flat_grad)-1, 10000, dtype=int)
        flat_grad = flat_grad[indices]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(flat_grad)), flat_grad, 'b-', alpha=0.7, linewidth=1)
    
    ax.set_xlabel('Parameter Index')
    ax.set_ylabel('Gradient Value')
    ax.set_title(f'{name} Values (Epoch {epoch})')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1e-3, 1e-3) # Fixed limits as in original code
    
    plt.tight_layout()
    experiment.log_figure(name, fig, step=epoch)
    plt.close(fig)


def log_gradient_histograms(name: str, grad_tensor: torch.Tensor, experiment: Any, epoch: int):
    flat_grad = grad_tensor.detach().cpu().flatten().numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.hist(flat_grad, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Gradient Value')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name} Histogram')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5e-4, 5e-4)
    
    plt.tight_layout()
    experiment.log_figure(name, fig, step=epoch)
    plt.close(fig)


def log_connectivity_heatmap(model: nn.Module, experiment: Any, epoch: int):
    target_layer = None
    for mod in model.modules():
        if isinstance(mod, ScaledNet):
            target_layer = mod
            break
    
    if target_layer is None: return

    weight_mask = (target_layer.weight.data != 0).float().cpu().numpy()
    
    if weight_mask.shape[0] > 100 or weight_mask.shape[1] > 100:
        h, w = min(weight_mask.shape[0], 100), min(weight_mask.shape[1], 200)
        weight_mask = weight_mask[:h, :w]
        title_suffix = f" (Top-Left {h}x{w} slice)"
    else:
        title_suffix = ""

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(weight_mask, cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
    ax.set_title(f"Connectivity Mask {title_suffix}\n(Black=Active, White=Pruned)")
    ax.set_xlabel("Input Features (Parents)")
    ax.set_ylabel("Output Neurons (Children)")
    
    experiment.log_figure("connectivity_mask", fig, step=epoch)
    plt.close(fig)


def log_weight_distributions_compare(model: nn.Module, experiment: Any, epoch: int):
    target_layer = None
    for mod in model.modules():
        if isinstance(mod, ScaledNet):
            target_layer = mod
            break
            
    if target_layer is None: return

    main_weights = target_layer.weight.data.flatten()
    main_weights = main_weights[main_weights != 0].cpu().numpy()
    
    sub_weights = []
    for sub in target_layer.sub_linears:
        w = sub.weight.data.flatten()
        w = w[w != 0]
        sub_weights.append(w)
    
    if sub_weights:
        sub_weights = torch.cat(sub_weights).cpu().numpy()
    else:
        sub_weights = np.array([])

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(main_weights, bins=100, alpha=0.5, label='Main Projection', density=True, color='blue', range=(-0.1, 0.1))
    
    if len(sub_weights) > 0:
        ax.hist(sub_weights, bins=100, alpha=0.5, label='SubLinear Layers', density=True, color='orange', range=(-0.1, 0.1))

    ax.set_title(f"Weight Value Distribution (Non-Zero only) - Epoch {epoch}")
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    experiment.log_figure("weight_dist_comparison", fig, step=epoch)
    plt.close(fig)


def log_gradient_flow_bar(model: nn.Module, experiment: Any, epoch: int):
    layers = []
    grads = []
    
    for name, module in model.named_modules():
        if isinstance(module, ScaledNet) and module.weight.grad is not None:
            layers.append(f"{name} (Main)")
            g = module.weight.grad.abs()
            mean_grad = g[g > 0].mean().item() if g.sum() > 0 else 0
            grads.append(mean_grad)
            
            for i, sub in enumerate(module.sub_linears):
                if sub.weight.grad is not None:
                    layers.append(f"{name}.sub[{i}]")
                    g = sub.weight.grad.abs()
                    mean_grad = g[g > 0].mean().item() if g.sum() > 0 else 0
                    grads.append(mean_grad)

    if not layers: return

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(layers))
    
    ax.barh(y_pos, grads, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layers)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Mean Absolute Gradient (Non-Zero)')
    ax.set_title(f'Gradient Flow Check - Epoch {epoch}')
    
    experiment.log_figure("gradient_flow", fig, step=epoch)
    plt.close(fig)


def log_sparsity_stability(model: nn.Module, experiment: Any, epoch: int):
    logs = {}
    for name, module in model.named_modules():
        if isinstance(module, (ScaledNet, SubLinear, LowRankAdapter)):
            total = module.weight.numel()
            zeros = (module.weight.data == 0).sum().item()
            
            abs_w = module.weight.data.abs()
            near_zeros = ((abs_w > 0) & (abs_w < 1e-9)).sum().item()
            
            logs[f"{name}_strict_sparsity"] = zeros / total
            logs[f"{name}_zombie_ratio"] = near_zeros / total

    experiment.log_metrics(logs, epoch=epoch)


def get_param_counts(model: nn.Module) -> Tuple[int, int]:
    """Returns (total_params, non_zero_params)"""
    total = 0
    non_zero = 0
    for module in model.modules():
        if isinstance(module, (ScaledNet, SubLinear, LowRankAdapter, nn.Linear)):
            for param in module.parameters(recurse=False):
                total += param.numel()
                non_zero += (param.data != 0).sum().item()
    return total, non_zero
