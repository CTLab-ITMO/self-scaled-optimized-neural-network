import time
import copy
from typing import Callable, Dict, Tuple, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error

from ssonn.model.model import ScaledNet
from ssonn.metrics.edge_finder import EdgeFinder


def apply_sparsity_mask(model: nn.Module):
    for module in model.modules():
        if isinstance(module, ScaledNet):
            def get_mask_hook(param):
                def hook(grad):
                    mask = (param.data != 0).float()
                    return grad * mask
                return hook

            if hasattr(module, 'weight') and module.weight.requires_grad:
                module.weight.register_hook(get_mask_hook(module.weight))
            

def train_one_epoch(
    model: nn.Module, 
    train_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    criterion: Callable, 
    device: str
) -> Tuple[float, float]:
    
    t0 = time.time()
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
    train_time = time.time() - t0
    return train_loss, train_time

def eval_one_epoch(
    model: nn.Module, 
    criterion: Callable, 
    val_loader: DataLoader, 
    task_type: str, 
    device: str
) -> Tuple[float, float]:
    
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
    expansion_happened = False
    total_new_edges = 0

    for layer_name, threshold in thresholds.items():
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
        else:
            continue

        if isinstance(layer, ScaledNet):
            parents, children = edge_finder.choose_edges_threshold(
                model, layer, threshold, embed=False
            )
            
            if len(parents) > 0:
                print(f"[Expansion] Layer '{layer_name}': Adding sub-layer for {len(parents)} edges.")
                layer.put_sub_linear(parents, children)
                expansion_happened = True
                total_new_edges += len(parents)
    
    return expansion_happened, total_new_edges

def prune_model(
    model: nn.Module, 
    edge_finder: EdgeFinder, 
    thresholds: Dict[str, float]
) -> int:
    pruned_count = 0
    for layer_name, threshold in thresholds.items():
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
        else:
            continue

        if isinstance(layer, ScaledNet):
            metric = edge_finder.calculate_edge_metric_for_dataloader(
                model=model, layer=layer, to_normalise=True, embed=False
            )
            
            current_nonzero = layer.weight.data.abs() > 1e-9
            mask_to_prune = (metric < threshold) & current_nonzero
            
            count = mask_to_prune.sum().item()
            if count > 0:
                with torch.no_grad():
                    layer.weight.data[mask_to_prune] = 0.0
                pruned_count += count
                
    if pruned_count > 0:
        print(f"Pruned: {pruned_count} weights.")

    return pruned_count


def train_ssonn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: Callable,
    optimizer_factory: Callable[[nn.Module], optim.Optimizer],
    hyperparams: Dict[str, Any],
    device: str
) -> nn.Module:
    """
    Main training loop for SSONN with dynamic expansion and pruning.
    
    Args:
        model: The model containing ScaledNet layers.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation/metrics.
        criterion: Loss function.
        optimizer_factory: A function that takes the model and returns an optimizer.
                           Required because model parameters change during expansion.
        hyperparams: Dictionary of hyperparameters.
        device: 'cpu' or 'cuda'.
    """
    model = model.to(device)
    optimizer = optimizer_factory(model)
    
    apply_sparsity_mask(model)

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
        t_loss, t_time = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_metric = eval_one_epoch(model, criterion, val_loader, hyperparams["task_type"], device)
        
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        print(f"Epoch {epoch+1}/{hyperparams['num_epochs']} | "
              f"Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f} | "
              f"Metric: {v_metric:.4f}")

        if epoch - last_expansion_epoch > hyperparams.get("min_epochs_between_expansions", 5):
            window = hyperparams.get("plateau_window_size", 5)
            if len(train_losses) >= window:
                recent = train_losses[-window:]
                avg_change = sum(abs(recent[i] - recent[i-1]) for i in range(1, len(recent))) / (window - 1)
                
                if avg_change < hyperparams.get("plateau_threshold", 0.01):
                    print(f"   -> Plateau detected (change {avg_change:.5f}). Checking expansion...")
                    
                    expanded, num_new = expand_model(
                        model, ef, hyperparams["expansion_thresholds"]
                    )
                    
                    if expanded:
                        print(f"   -> Model expanded. Re-initializing optimizer.")
                        optimizer = optimizer_factory(model)
                        apply_sparsity_mask(model)
                        
                        last_expansion_epoch = epoch
        
        if epoch > hyperparams.get("prune_after_epochs", 10):
            prune_model(model, ef, hyperparams.get("pruning_thresholds", {}))

    print("--- Training Finished ---")
    return model

def convert_to_scaled_net(model: nn.Module, layers_to_convert: list, device="cpu") -> nn.Module:
    new_model = copy.deepcopy(model)
    
    layer_map = {}
    for name, module in model.named_children():
        layer_map[module] = name
        
    for layer in layers_to_convert:
        if layer in layer_map:
            name = layer_map[layer]
            print(f"Converting layer '{name}' to ScaledNet...")
            
            scaled_layer = ScaledNet(
                layer.weight.data, 
                layer.bias.data if layer.bias is not None else torch.zeros(layer.out_features),
                device=device
            )
            setattr(new_model, name, scaled_layer)
        else:
            print(f"Warning: Layer {layer} not found in model children.")
            
    return new_model.to(device)
