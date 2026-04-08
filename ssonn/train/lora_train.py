from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ssonn.model.expandable_linear import SelfExpandingNetwork
from ssonn.train.cosine_full_cycle import CosineFullCycleLR

OptimizerFactory = Callable[[nn.Module, Dict[str, Any]], optim.Optimizer]
SchedulerFactory = Callable[[optim.Optimizer, Dict[str, Any]], Any]


def default_expandable_optimizer_factory(
    model: nn.Module, hp: Dict[str, Any]
) -> optim.Optimizer:
    return optim.SGD(
        model.parameters(),
        lr=float(hp.get("optimizer_initial_lr", 0.0)),
        momentum=float(hp.get("momentum", 0.9)),
        weight_decay=float(hp.get("weight_decay", 0.0)),
    )


def default_expandable_scheduler_factory(
    optimizer: optim.Optimizer, hp: Dict[str, Any]
) -> CosineFullCycleLR:
    return CosineFullCycleLR(
        optimizer,
        float(hp["max_lr"]),
        int(hp["epochs_per_cycle"]),
    )


def _params_for_comet(h: Dict[str, Any]) -> Dict[str, Any]:
    """Только сериализуемые в Comet значения."""
    out: Dict[str, Any] = {}
    for k, v in h.items():
        if isinstance(v, (bool, int, float, str)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(
            isinstance(x, (bool, int, float, str)) for x in v
        ):
            out[k] = list(v)
    return out


@torch.no_grad()
def accuracy_on_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        pred = model(data).argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.size(0)
    return correct / max(total, 1)


def train_expandable_cycles(
    model: SelfExpandingNetwork,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    hyperparams: Dict[str, Any],
    experiment: Any = None,
    use_log_tool: bool = True,
    optimizer_factory: Optional[OptimizerFactory] = None,
    scheduler_factory: Optional[SchedulerFactory] = None,
) -> SelfExpandingNetwork:
    """
    Циклы: планировщик LR → на последней эпохе сбор статистики градиентов → тест →
    expand_all → переинициализация He.

    hyperparams:
        max_lr, epochs_per_cycle, num_cycles,
        threshold (для expansion_mode 'threshold', опц. 0.8),
        expansion_mode (опц., 'threshold' | 'top_fraction'),
        expansion_top_fraction (для top_fraction, опц. 0.3 — доля in_features),
        low_dim_factor (опц., 10), momentum (опц., 0.9),
        optimizer_initial_lr (опц., 0.0), weight_decay (опц., 0.0).

    optimizer_factory(model, hp) — новый оптимизатор в начале каждого цикла (по умолчанию SGD).
    scheduler_factory(optimizer, hp) — объект с методом step(epoch: int) (по умолчанию CosineFullCycleLR).

    verbose (опц., True) — печать loss / train acc / val acc каждую эпоху.
    """
    epochs_per_cycle = int(hyperparams["epochs_per_cycle"])
    num_cycles = int(hyperparams["num_cycles"])
    threshold = float(hyperparams.get("threshold", 0.8))
    expansion_mode = str(hyperparams.get("expansion_mode", "threshold"))
    expansion_top_fraction = float(hyperparams.get("expansion_top_fraction", 0.3))
    low_dim_factor = int(hyperparams.get("low_dim_factor", 10))
    verbose = bool(hyperparams.get("verbose", True))

    opt_factory = optimizer_factory or default_expandable_optimizer_factory
    sched_factory = scheduler_factory or default_expandable_scheduler_factory

    if use_log_tool and experiment is not None:
        experiment.log_parameters(_params_for_comet(hyperparams))

    global_step = 0
    model = model.to(device)

    if verbose:
        print("--- SelfExpandingNetwork (expandable) training ---")

    for cycle in range(num_cycles):
        optimizer = opt_factory(model, hyperparams)
        scheduler = sched_factory(optimizer, hyperparams)

        for epoch in range(epochs_per_cycle):
            model.train()
            collect_stats = epoch == epochs_per_cycle - 1

            if collect_stats:
                for layer in model.get_all_expandable_layers():
                    layer.init_statistics()
                    layer.register_statistics_hook()

            epoch_loss = 0.0
            n_batches = 0
            train_correct = 0
            train_total = 0
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                if collect_stats:
                    data.requires_grad_(True)

                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1
                pred = output.argmax(dim=1)
                train_correct += (pred == target).sum().item()
                train_total += target.size(0)

            if collect_stats:
                for layer in model.get_all_expandable_layers():
                    layer.remove_statistics_hook()

            scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]["lr"]
            avg_loss = epoch_loss / max(n_batches, 1)
            train_acc = train_correct / max(train_total, 1)
            val_acc = accuracy_on_loader(model, test_loader, device)
            total_params = sum(p.numel() for p in model.parameters())
            n_blocks = sum(
                len(el.low_rank_blocks) for el in model.get_all_expandable_layers()
            )

            if verbose:
                print(
                    f"Cycle {cycle + 1}/{num_cycles} | "
                    f"Epoch {epoch + 1}/{epochs_per_cycle} "
                    f"(step {global_step + 1}) | "
                    f"T-Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Acc: {val_acc:.4f} | LR: {current_lr:.2e} | "
                    f"Params: {total_params} | Blocks: {n_blocks}"
                )

            if use_log_tool and experiment is not None:
                experiment.log_metrics(
                    {
                        "train loss": avg_loss,
                        "train accuracy": train_acc,
                        "val accuracy": val_acc,
                        "test accuracy": val_acc,
                        "lr": current_lr,
                        "cycle": float(cycle),
                        "epoch_in_cycle": float(epoch),
                        "total params": float(total_params),
                        "low_rank_blocks": float(n_blocks),
                    },
                    step=global_step,
                )
            global_step += 1

        params_before = sum(p.numel() for p in model.parameters())
        blocks_before = sum(
            len(el.low_rank_blocks) for el in model.get_all_expandable_layers()
        )

        if verbose:
            print(
                f"   -> Expanding (cycle {cycle + 1}/{num_cycles}): "
                f"params {params_before}, low-rank blocks {blocks_before}"
            )

        model.expand_all(
            threshold=threshold,
            low_dim_factor=low_dim_factor,
            expansion_mode=expansion_mode,
            expansion_top_fraction=expansion_top_fraction,
        )
        params_after = sum(p.numel() for p in model.parameters())
        blocks_after = sum(
            len(el.low_rank_blocks) for el in model.get_all_expandable_layers()
        )
        model.reinitialize_all_parameters()

        if verbose:
            print(
                f"   -> After expansion: params {params_after}, "
                f"new blocks +{blocks_after - blocks_before}, "
                f"reinitialized He"
            )

        if use_log_tool and experiment is not None:
            experiment.log_metrics(
                {
                    "expansion params delta": float(params_after - params_before),
                    "expansion blocks delta": float(blocks_after - blocks_before),
                    "total params after expansion": float(params_after),
                    "low_rank_blocks after expansion": float(blocks_after),
                },
                step=global_step - 1,
            )
            experiment.log_metric(
                "expansion event",
                float(blocks_after - blocks_before),
                step=global_step - 1,
            )

    if verbose:
        print("--- SelfExpandingNetwork training finished ---")

    return model
