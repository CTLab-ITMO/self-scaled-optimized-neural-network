import math
from typing import List, Sequence

import torch
import torch.nn as nn


class ExpandableLinear(nn.Module):
    """Линейный слой с динамическим расширением за счёт low-rank блоков (Linear → ReLU → Linear)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.low_rank_blocks = nn.ModuleList()
        self.used_indices: List[List[int]] = []
        self.register_buffer("used_mask", torch.zeros(in_features, dtype=torch.bool))

        self.sum_grad: torch.Tensor | None = None
        self.sum_sq_grad: torch.Tensor | None = None
        self.count = 0
        self._hook_handle = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        for block, indices in zip(self.low_rank_blocks, self.used_indices):
            x_sel = x[:, indices]
            out = out + block(x_sel)
        return out

    def expand(self, new_indices: Sequence[int], low_dim_factor: int = 10) -> None:
        """Добавляет low-rank блок по заданным индексам входных признаков."""
        if not new_indices:
            return
        n_selected = len(new_indices)
        n_low = max(1, n_selected // low_dim_factor)
        fc1 = ExpandableLinear(n_selected, n_low, bias=True)
        fc2 = ExpandableLinear(n_low, self.out_features, bias=True)
        block = nn.Sequential(fc1, nn.ReLU(), fc2)
        dev = self.linear.weight.device
        dtype = self.linear.weight.dtype
        block = block.to(device=dev, dtype=dtype)
        self.low_rank_blocks.append(block)
        self.used_indices.append(list(new_indices))
        idx = torch.as_tensor(new_indices, device=self.used_mask.device, dtype=torch.long)
        self.used_mask[idx] = True

    def init_statistics(self) -> None:
        device = self.linear.weight.device
        self.sum_grad = torch.zeros(self.in_features, device=device)
        self.sum_sq_grad = torch.zeros(self.in_features, device=device)
        self.count = 0

    def clear_statistics(self) -> None:
        self.sum_grad = None
        self.sum_sq_grad = None
        self.count = 0

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        if grad_input[0] is not None:
            grad = grad_input[0]
            assert self.sum_grad is not None and self.sum_sq_grad is not None
            self.sum_grad = self.sum_grad + grad.sum(dim=0)
            self.sum_sq_grad = self.sum_sq_grad + (grad ** 2).sum(dim=0)
            self.count += grad.size(0)

    def register_statistics_hook(self) -> None:
        self._hook_handle = self.register_full_backward_hook(self._backward_hook)

    def remove_statistics_hook(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None


class SelfExpandingNetwork(nn.Module):
    """Последовательность ExpandableLinear (плоский вход → выход)."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ExpandableLinear(input_dim, output_dim, bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

    def get_all_expandable_layers(self) -> List[ExpandableLinear]:
        layers: List[ExpandableLinear] = []
        for module in self.modules():
            if isinstance(module, ExpandableLinear):
                layers.append(module)
        return layers

    def expand_all(
        self,
        threshold: float = 0.8,
        low_dim_factor: int = 10,
        expansion_mode: str = "threshold",
        expansion_top_fraction: float = 0.3,
    ) -> None:
        """
        Расширяет каждый ExpandableLinear по дисперсии градиентов входа.

        expansion_mode:
            - "threshold": нормализованная дисперсия > threshold (как раньше).
            - "top_fraction": брать top-k признаков по дисперсии, k = min(ceil(fraction * in_features), число свободных).
        """
        if expansion_mode not in ("threshold", "top_fraction"):
            raise ValueError(
                f"expansion_mode must be 'threshold' or 'top_fraction', got {expansion_mode!r}"
            )

        layers = self.get_all_expandable_layers()
        for layer in layers:
            if layer.count == 0 or layer.sum_grad is None or layer.sum_sq_grad is None:
                continue

            mean_grad = layer.sum_grad / layer.count
            mean_sq_grad = layer.sum_sq_grad / layer.count
            variance = mean_sq_grad - mean_grad ** 2
            variance = torch.clamp(variance, min=0.0)

            available = ~layer.used_mask
            if not bool(available.any()):
                continue

            n_available = int(available.sum().item())
            if n_available == 0:
                continue

            if expansion_mode == "top_fraction":
                frac = float(expansion_top_fraction)
                if not (0.0 < frac <= 1.0):
                    raise ValueError(
                        f"expansion_top_fraction must be in (0, 1], got {frac}"
                    )
                n_select = int(frac * layer.in_features)
                n_select = max(0, min(n_select, n_available))
                if n_select == 0:
                    continue
                var_copy = variance.clone()
                var_copy[~available] = float("-inf")
                max_var = var_copy.max()
                if not torch.isfinite(max_var) or max_var <= 0:
                    continue
                _, topk_flat = torch.topk(var_copy, n_select)
                new_indices = topk_flat.detach().cpu().tolist()
                layer.expand(new_indices, low_dim_factor=low_dim_factor)
                continue

            # threshold mode
            var_available = variance[available]
            max_var = var_available.max()
            if max_var <= 0:
                continue

            normalized_var = var_available / max_var
            selected_available_indices = torch.where(normalized_var > threshold)[0]
            if selected_available_indices.numel() == 0:
                continue

            global_indices = torch.where(available)[0][selected_available_indices]
            new_indices = global_indices.detach().cpu().tolist()
            layer.expand(new_indices, low_dim_factor=low_dim_factor)

    def reinitialize_all_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(module.bias, -bound, bound)
