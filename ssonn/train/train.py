import copy
from typing import Callable, Dict, Tuple, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error

from ssonn.model.model import ScaledNet, SubLinear
from ssonn.metrics.edge_finder import EdgeFinder
from .scheduler import SSONNPlateauScheduler, SSONNWarmupScheduler
from .logging import *


def apply_sparsity_mask(model: nn.Module) -> None:
    """
    Registers backward hooks to enforce sparsity on dense tensors.
    """
    for module in model.modules():
        if isinstance(module, (ScaledNet, SubLinear)):

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
        device: str,
        max_norm: float = 1.0  # Добавляем параметр для порога клиппинга
) -> Tuple[float, float]:
    """
    Performs one epoch of training with gradient clipping.

    Returns:
        Tuple[float, float]: Average training loss and accuracy.
    """
    model.train()
    train_loss = 0.0
    all_targets = []
    all_preds = []

    pbar = tqdm(train_loader, desc="Train", leave=False)

    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # --- Gradient Clipping ---
        # Важно делать это ПОСЛЕ loss.backward(), но ДО optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # -------------------------

        optimizer.step()

        train_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    train_loss /= len(train_loader)
    metric = accuracy_score(all_targets, all_preds)

    return train_loss, metric
    

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


def prune_model(
        model: nn.Module,
        optimizer: optim.Optimizer,
        edge_finder: EdgeFinder,
        thresholds: Dict[str, float]
) -> int:
    """
    Zeros out weights AND clears optimizer state to prevent resurrection.
    """
    pruned_count = 0

    for layer_name, threshold in thresholds.items():
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
        else:
            continue

        if isinstance(layer, ScaledNet):
            # 1. Compute metrics (ScaledNet)
            metric_exp = edge_finder.calculate_edge_metric_for_dataloader(
                model=model, layer=layer, to_normalise=False, embed=False
            )

            # 2. Compute metrics (SubLinear), if exists
            has_sub_layers = len(layer.sub_linears) > 0
            if has_sub_layers:
                metric_emb = edge_finder.calculate_edge_metric_for_dataloader(
                    model=model, layer=layer, to_normalise=False, embed=True
                )

                flat_exp = metric_exp.flatten()
                flat_emb = metric_emb.flatten()
                combined = torch.cat([flat_emb, flat_exp])
                min_v, max_v = combined.min(), combined.max()
                norm = (combined - min_v) / (max_v - min_v + 1e-8)

                global_mask = norm < threshold

                num_emb = flat_emb.numel()
                mask_emb = global_mask[:num_emb].view_as(metric_emb)
                mask_exp = global_mask[num_emb:].view_as(metric_exp)

                target_sub = layer.sub_linears[-1]
                # Only non-zero params
                nonzero_emb = target_sub.weight.data.abs() > 1e-9
                final_mask_emb = mask_emb & nonzero_emb

                if final_mask_emb.sum() > 0:
                    with torch.no_grad():
                        target_sub.weight.data[final_mask_emb] = 0.0

                        if target_sub.weight in optimizer.state:
                            state = optimizer.state[target_sub.weight]
                            if 'exp_avg' in state:
                                state['exp_avg'][final_mask_emb] = 0.0
                            if 'exp_avg_sq' in state:
                                state['exp_avg_sq'][final_mask_emb] = 0.0
                    pruned_count += final_mask_emb.sum().item()

            else:
                min_v, max_v = metric_exp.min(), metric_exp.max()
                norm = (metric_exp - min_v) / (max_v - min_v + 1e-8)
                mask_exp = norm < threshold

            nonzero_exp = layer.weight.data.abs() > 1e-9
            final_mask_exp = mask_exp & nonzero_exp

            if final_mask_exp.sum() > 0:
                with torch.no_grad():
                    layer.weight.data[final_mask_exp] = 0.0

                    if layer.weight in optimizer.state:
                        state = optimizer.state[layer.weight]
                        if 'exp_avg' in state:
                            state['exp_avg'][final_mask_exp] = 0.0
                        if 'exp_avg_sq' in state:
                            state['exp_avg_sq'][final_mask_exp] = 0.0

                pruned_count += final_mask_exp.sum().item()

    if pruned_count > 0:
        print(f"[Pruning] Total weights zeroed out: {pruned_count}")

    return pruned_count


def build_preserved_param_groups(model: nn.Module, old_optimizer: optim.Optimizer) -> List[Dict[str, Any]]:
    """
    Создает группы параметров для нового оптимизатора.
    Новый слой получает новую группу.
    """
    id_to_group_idx = {} # map из id параметра к индексу его группы в старом оптимизаторе
    for idx, group in enumerate(old_optimizer.param_groups):
        for p in group['params']:
            id_to_group_idx[id(p)] = idx

    new_params_ids = set()
    new_params = []
    for module in model.modules():
        if isinstance(module, ScaledNet) and len(module.sub_linears) > 0:
            last_sub = module.sub_linears[-1]
            for p in last_sub.parameters():
                new_params.append(p)
                new_params_ids.add(id(p))

    num_old_groups = len(old_optimizer.param_groups)
    groups_params = [[] for _ in range(num_old_groups)]

    for p in model.parameters():
        if id(p) in new_params_ids:
            continue

        if id(p) in id_to_group_idx:
            g_idx = id_to_group_idx[id(p)]
            groups_params[g_idx].append(p)
        else:
            groups_params[0].append(p)

    param_groups_final = []  # список групп для отимизатора с torch.nn.Parameter - lr мапой
    for i in range(num_old_groups):
        old_lr = old_optimizer.param_groups[i]['lr']
        if len(groups_params[i]) > 0:
            param_groups_final.append({'params': groups_params[i], 'lr': old_lr})

    param_groups_final.append({'params': new_params, 'lr': 0.0})

    return param_groups_final


def train_ssonn(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    finder_loader: DataLoader,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer_factory: Callable[[nn.Module], optim.Optimizer],
    hyperparams: Dict[str, Any],
    device: str,
    experiment: Any = None,
    use_log_tool: bool = True
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
    ef = EdgeFinder(
        metric=hyperparams["edge_importance_metric"],
        dataloader=finder_loader,
        device=device,
        aggregation_mode=hyperparams.get("edge_score_aggregation", "mean"),
        max_to_replace=hyperparams.get("max_new_edges_per_expansion", None),
    )
    scheduler = SSONNPlateauScheduler(hyperparams)
    sched = CosineAnnealingLR(optimizer, T_max=hyperparams["t_max"], eta_min=0, last_epoch=-1)
    warmup_scheduler = None

    target_layer = None
    target_layer_name = "unknown"
    for name, mod in model.named_modules():
        if isinstance(mod, ScaledNet):
            target_layer = mod
            target_layer_name = name
            break

    train_losses = []
    val_losses = []

    print("--- Starting SSONN Training ---")

    for epoch in range(hyperparams["num_epochs"]):
        # 1. Train Step
        t_loss, t_metric = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.update_metrics(t_loss)

        if warmup_scheduler is not None:
            warmup_scheduler.step()
            if warmup_scheduler.current_epoch >= warmup_scheduler.warmup_epochs:
                warmup_scheduler = None

        # 2. Eval Step
        v_loss, v_metric = eval_one_epoch(model, criterion, val_loader, hyperparams["task_type"], device)

        train_losses.append(t_loss)
        val_losses.append(v_loss)

        # 3. Scale Step
        # TODO: move to scheduler module
        if ((epoch - hyperparams["t_max"]) % (2 * hyperparams["t_max"])) == 0 and epoch != 0:
            print(f"   -> Action: Expanding Model.")
            expanded, num_new = expand_model(model, ef, hyperparams["expansion_thresholds"])
          
            if expanded:
                print(f"   -> Re-initializing ScaledNet weights.")
                model.apply(weights_init)

                print(f"   -> Re-initializing optimizer.")
                current_lr = optimizer.param_groups[0]['lr']
              
                optimizer = optimizer_factory(
                    model, 
                    current_lr=current_lr, 
                    base_lr=hyperparams['learning_rate']
                )
              
                sched = CosineAnnealingLR(
                    optimizer, 
                    T_max=hyperparams["t_max"], 
                    eta_min=0, 
                    last_epoch=epoch
                )
                apply_sparsity_mask(model)                
                if use_log_tool and experiment:
                    experiment.log_metric("expansion event", num_new, epoch=epoch)
            else:
                print(f"   -> Expansion attempted but no edges selected.")

        sched.step()
        
        # 4. Pruning Logic
        # if epoch > hyperparams.get("prune_after_epochs", 999999):
        #     prune_model(model, optimizer, ef, hyperparams.get("pruning_thresholds", {}))
        # if pruned > 0 and use_log_tool and experiment:
        #      experiment.log_metric("pruned weights", pruned, epoch=epoch)

        # 5. Calculate Params stats, Logging (Comet ML)
        total_params, non_zero_params = get_param_counts(model)

        lrs = [g['lr'] for g in optimizer.param_groups]
        lr_str = " | ".join([f"{lr:.1e}" for lr in lrs])

        print(f"Epoch {epoch + 1}/{hyperparams['num_epochs']} | "
              f"T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f} | Acc: {v_metric:.4f} | "
              f"LRs: [{lr_str}] | Params: {non_zero_params}/{total_params}")

        if use_log_tool and experiment is not None:
            logs = {
                "train loss": t_loss,
                "val loss": v_loss,
                "val accuracy": v_metric,
                "train accuracy": t_metric,
                "lr": optimizer.param_groups[0]["lr"],
                "total params": total_params,
                "params amount": non_zero_params,
                "sparsity": 1.0 - (non_zero_params / total_params),
            }

            if target_layer:
                metric_main = ef.calculate_edge_metric_for_dataloader(
                        model, target_layer, to_normalise=False, embed=False)
                logs[f"choose_metric/{target_layer_name}_main"] = float(metric_main.mean().item())
                if len(target_layer.sub_linears) > 0:
                    metric_sub = ef.calculate_edge_metric_for_dataloader(
                        model, target_layer, to_normalise=False, embed=True)
                    logs[f"choose_metric/{target_layer_name}_sub"] = metric_sub.mean().item()

            experiment.log_metrics(logs, step=epoch)

            # if target_layer and target_layer.weight.grad is not None:
            #     log_gradient_values("main_layer_grads", target_layer.weight.grad, experiment, epoch)
            #     log_gradient_histograms("main_layer_grads_hist", target_layer.weight.grad, experiment, epoch)
            #     # log_gradient_image(target_layer, experiment, epoch)

    
        # TODO: add flag for model saving 
        if epoch % 50 == 0:
            torch.save(model, f"ssonn_{epoch}.pth")

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


def weights_init(m):
    if isinstance(m, ScaledNet):
        m.reinitialize_parameters()
    
    elif isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
            
    elif isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
