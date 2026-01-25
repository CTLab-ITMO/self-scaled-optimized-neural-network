import time
import copy
from typing import Callable, Dict, Tuple, Any, List, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error

from ssonn.model.model import ScaledNet
from ssonn.metrics.edge_finder import EdgeFinder


def apply_sparsity_mask(model: nn.Module) -> None:
    """
    Registers backward hooks to enforce sparsity on dense tensors.
    """
    for module in model.modules():
        if isinstance(module, ScaledNet):
            def get_mask_hook(param: nn.Parameter) -> Callable[[torch.Tensor], torch.Tensor]:
                def hook(grad: torch.Tensor) -> torch.Tensor:
                    mask = (param.data != 0).float()
                    return grad * mask
                return hook

            # Register the hook if the weight parameter exists and requires gradients
            if hasattr(module, 'weight') and module.weight.requires_grad:
                module.weight.register_hook(get_mask_hook(module.weight))


def train_one_epoch(
    model: nn.Module, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
    device: str
) -> float:
    """
    Performs one epoch of training.

    Returns:
        Tuple[float, float]: Average training loss and the time taken for the epoch.
    """
    model.train()
    train_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Train", leave=False)
    
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    train_loss /= len(train_loader)
    return train_loss


def eval_one_epoch(
    model: nn.Module, 
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], 
    val_loader: DataLoader, 
    task_type: str, 
    device: str
) -> Tuple[float, float]:
    """
    Evaluates the model on the validation set.

    Args:
        task_type: 'classification' (uses Accuracy) or 'regression' (uses MSE).

    Returns:
        Tuple[float, float]: Average validation loss and the calculated metric.
    """
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            if task_type == "classification":
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = outputs

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    val_loss /= len(val_loader)

    if task_type == "classification":
        metric = accuracy_score(all_targets, all_preds)
    else:
        metric = mean_squared_error(all_targets, all_preds)

    return val_loss, metric


def expand_model(
    model: nn.Module, 
    edge_finder: EdgeFinder, 
    thresholds: Dict[str, float]
) -> Tuple[bool, int]:
    """
    Checks if layers need expansion and applies the SSONN expansion operation.

    Returns:
        Tuple[bool, int]: (True if architecture changed, Number of new edges added).
    """
    expansion_happened = False
    total_new_edges = 0

    for layer_name, threshold in thresholds.items():
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
        else:
            continue

        if isinstance(layer, ScaledNet):
            # Identify high-importance edges to expand
            # embed=False targets the main projection matrix
            parents, children = edge_finder.choose_edges_threshold(
                model, layer, threshold, embed=False
            )
            
            if len(parents) > 0:
                print(f"[Expansion] Layer '{layer_name}': Adding sub-layer for {len(parents)} edges.")
                # Apply structural expansion
                layer.put_sub_linear(parents, children)
                expansion_happened = True
                total_new_edges += len(parents)
    
    return expansion_happened, total_new_edges


# TODO: add pruning on SubLinear
def prune_model(
    model: nn.Module, 
    edge_finder: EdgeFinder, 
    thresholds: Dict[str, float]
) -> int:
    """
    Zeros out weights that fall below the importance threshold.

    Returns:
        int: Total number of weights pruned.
    """
    pruned_count = 0
    for layer_name, threshold in thresholds.items():
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
        else:
            continue

        if isinstance(layer, ScaledNet):
            # Calculate importance metrics for all weights
            metric = edge_finder.calculate_edge_metric_for_dataloader(
                model=model, layer=layer, to_normalise=True, embed=False
            )
            
            # Pruning Condition: (Metric < Threshold) AND (Weight is currently active)
            # We check > 1e-9 to avoid re-pruning already zeroed weights
            current_nonzero = layer.weight.data.abs() > 1e-9
            mask_to_prune = (metric < threshold) & current_nonzero
            
            count = mask_to_prune.sum().item()
            if count > 0:
                with torch.no_grad():
                    # Set weights to exactly 0.0
                    layer.weight.data[mask_to_prune] = 0.0
                pruned_count += count
                
    if pruned_count > 0:
        print(f"[Pruning] Zeroed out {pruned_count} weights.")

    return pruned_count


def train_ssonn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer_factory: Callable[[nn.Module], optim.Optimizer],
    hyperparams: Dict[str, Any],
    device: str
) -> nn.Module:
    """
    Manupulate the SSONN training lifecycle with dynamic expansion and pruning.

    This function handles:
    1. Standard training/evaluation loops.
    2. Plateau detection to trigger model expansion.
    3. Structural updates (Expansion/Pruning).
    4. Optimizer re-initialization (required when model parameters change).

    Args:
        model: The initial neural network.
        optimizer_factory: A function that returns a new optimizer instance for a given model.
                           This is crucial because `ScaledNet` creates new parameter tensors 
                           during expansion, invalidating the old optimizer state.
        hyperparams: Configuration dictionary.
    """
    model = model.to(device)
    # Initialize optimizer using the factory
    optimizer = optimizer_factory(model)
    
    # Register hooks to keep zeroed weights at zero
    apply_sparsity_mask(model)

    # Initialize EdgeFinder for importance scoring
    # TODO: add train_weight, train_architecture datasets splitting @shivan
    # TODO: move EdgeFinder init from train_ssonn to experiments
    ef = EdgeFinder(
        metric=hyperparams["edge_importance_metric"],
        dataloader=val_loader,
        device=device,
        aggregation_mode=hyperparams.get("edge_score_aggregation", "mean"),
        max_to_replace=hyperparams.get("max_new_edges_per_expansion", None),
    )

    train_losses = []
    val_losses = []
    last_expansion_epoch = 0

    print("--- Starting SSONN Training ---")
    
    for epoch in range(hyperparams["num_epochs"]):
        # 1. Train Step
        t_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # 2. Eval Step
        v_loss, v_metric = eval_one_epoch(model, criterion, val_loader, hyperparams["task_type"], device)
        
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        print(f"Epoch {epoch+1}/{hyperparams['num_epochs']} | "
              f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
              f"Metric: {v_metric:.4f}")

        # 3. Expansion Logic: Check for loss plateau
        if epoch - last_expansion_epoch > hyperparams.get("min_epochs_between_expansions", 5):
            window = hyperparams.get("plateau_window_size", 5)
            
            if len(train_losses) >= window:
                # Calculate average absolute change in loss over the window
                recent = train_losses[-window:]
                avg_change = sum(abs(recent[i] - recent[i-1]) for i in range(1, len(recent))) / (window - 1)
                
                # If loss has stabilized (plateau), attempt to expand capacity
                if avg_change < hyperparams.get("plateau_threshold", 0.01):
                    print(f"   -> Plateau detected (change {avg_change:.5f}). Checking expansion...")
                    
                    expanded, num_new = expand_model(
                        model, ef, hyperparams["expansion_thresholds"]
                    )
                    
                    if expanded:
                        print(f"   -> Model expanded. Re-initializing optimizer.")
                        # Architecture changed: Old optimizer references dead parameters.
                        # We must create a new optimizer for the updated model.
                        optimizer = optimizer_factory(model)
                        
                        # Re-apply hooks to the new parameter tensors
                        apply_sparsity_mask(model)
                        
                        last_expansion_epoch = epoch
        
        # 4. Pruning Logic
        if epoch > hyperparams.get("prune_after_epochs", 10):
            prune_model(model, ef, hyperparams.get("pruning_thresholds", {}))

    print("--- Training Finished ---")
    return model


def convert_to_scaled_net(
    model: nn.Module, 
    layers_to_convert: List[ScaledNet], 
    device: str
) -> nn.Module:
    """
    Replaces specified standard linear layers with SSONN `ScaledNet` layers.
    
    Useful for starting SSONN training from a pre-trained standard model.
    """
    new_model = copy.deepcopy(model)
    
    # Create a mapping from layer object to its name in the module
    layer_map = {}
    for name, module in model.named_children():
        layer_map[module] = name
        
    for layer in layers_to_convert:
        if layer in layer_map:
            name = layer_map[layer]
            print(f"Converting layer '{name}' to ScaledNet...")
            
            # Initialize ScaledNet with the weights of the original layer
            scaled_layer = ScaledNet(
                layer.weight.data, 
                layer.bias.data if layer.bias is not None else torch.zeros(layer.out_features),
                device=device
            )
            setattr(new_model, name, scaled_layer)
        else:
            print(f"Warning: Layer {layer} not found in model children.")
            
    return new_model.to(device)
