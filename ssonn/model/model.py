import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

class ScaledNet(nn.Module):
    """
    Implements the Self-Scaled Optimized Neural Network core module.
    """
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, device: str):
        super(ScaledNet, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(p=0.3)

        self.weight = nn.Parameter(weight.to(device))
        self.bias = nn.Parameter(bias.to(device))

        self.out_features, self.in_features = weight.shape

        self.sub_linears = nn.ModuleList()

    def reinitialize_parameters(self):
        with torch.no_grad():
            main_mask = (self.weight != 0).float()
            self.register_buffer('sparsity_mask', main_mask)

            nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
                
            self.weight.data.mul_(self.sparsity_mask)
            
            for sub_layer in self.sub_linears:
                sub_mask = (sub_layer.weight != 0).float()
                sub_layer.register_buffer('sparsity_mask', sub_mask)
                
                nn.init.kaiming_uniform_(sub_layer.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
                if sub_layer.bias is not None:
                    nn.init.zeros_(sub_layer.bias)
                
                sub_layer.weight.data.mul_(sub_layer.sparsity_mask)
                    
    def put_sub_linear(self, parents: torch.Tensor, children: torch.Tensor):
        parents = parents.to(self.device).long()
        children = children.to(self.device).long()
        chosen_dim = parents.shape[0]

        self.sub_linears.append(SubLinear(parents, self.in_features, self.device))

        old_weight_data = self.weight.data
        weight_cat = torch.zeros((self.out_features, chosen_dim), device=self.device)
        weight_cat[children, torch.arange(0, chosen_dim)] = old_weight_data[children, parents]
        
        old_weight_data[children, parents] = 0.
        new_weight_data = torch.cat([old_weight_data, weight_cat], dim=1)

        self.weight = nn.Parameter(new_weight_data)
        self.in_features += chosen_dim
        if hasattr(self, 'sparsity_mask'):
            delattr(self, 'sparsity_mask')

    def put_low_rank_adapter(
            self,
            parents: torch.Tensor,
            children: torch.Tensor,
            rank: Optional[int] = None,
            activation: bool = False,
            init_scale: float = 1e-8
    ) -> int:
        """
        Adds a LoRA-like bottleneck adapter for selected edges.

        The original projection is kept intact. The adapter adds a small
        parent -> bottleneck -> child path initialized with tiny non-zero
        weights, so the architecture grows while the initial function changes
        only negligibly.
        """
        parents = parents.to(self.device).long()
        children = children.to(self.device).long()
        if parents.numel() == 0:
            return 0

        unique_parents = torch.unique(parents)
        unique_children = torch.unique(children)
        max_rank = min(unique_parents.numel(), unique_children.numel())
        if max_rank == 0:
            return 0

        adapter_rank = max_rank if rank is None else min(rank, max_rank)
        adapter_rank = int(max(1, adapter_rank))

        old_weight_data = self.weight.data
        adapter_weight = torch.zeros(
            (adapter_rank, self.in_features),
            device=self.device,
            dtype=old_weight_data.dtype
        )
        adapter_weight[:, unique_parents] = torch.empty(
            (adapter_rank, unique_parents.numel()),
            device=self.device,
            dtype=old_weight_data.dtype
        ).uniform_(-init_scale, init_scale)

        output_weight = torch.zeros(
            (self.out_features, adapter_rank),
            device=self.device,
            dtype=old_weight_data.dtype
        )
        output_weight[unique_children, :] = torch.empty(
            (unique_children.numel(), adapter_rank),
            device=self.device,
            dtype=old_weight_data.dtype
        ).uniform_(-init_scale, init_scale)

        self.sub_linears.append(
            LowRankAdapter(adapter_weight, self.device, activation=activation)
        )
        self.weight = nn.Parameter(torch.cat([old_weight_data, output_weight], dim=1))
        self.in_features += adapter_rank
        if hasattr(self, 'sparsity_mask'):
            delattr(self, 'sparsity_mask')

        return adapter_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for sub_linear in self.sub_linears:
            x = self.dropout(sub_linear(x))

        effective_weight = self.weight * self.sparsity_mask if hasattr(self, 'sparsity_mask') else self.weight
        return F.linear(x, effective_weight, self.bias)


class SubLinear(nn.Module):
    """
    Represents an intermediate expansion layer.
    """
    def __init__(self, parents: torch.Tensor, input_dim: int, device: str):
        super(SubLinear, self).__init__()
        self.device = device
        output_dim = parents.shape[0]

        weight_data = torch.zeros((output_dim, input_dim), device=device)
        unique_parents = torch.unique(parents)
        
        temp_weights = torch.empty_like(weight_data)
        nn.init.kaiming_uniform_(temp_weights, mode='fan_in', nonlinearity='leaky_relu', a=0.01)
        
        weight_data[:, unique_parents] = temp_weights[:, unique_parents]
        
        self.weight = nn.Parameter(weight_data)
        
        self.bias = nn.Parameter(torch.zeros((output_dim,), device=device))

        self.act = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        effective_weight = self.weight * self.sparsity_mask if hasattr(self, 'sparsity_mask') else self.weight
        
        out = F.linear(x, effective_weight, self.bias)
        out = self.act(out)
        return torch.cat([x, out], dim=-1)


class LowRankAdapter(nn.Module):
    """
    Bottleneck expansion layer used for LoRA-like structural growth.
    """
    def __init__(self, weight: torch.Tensor, device: str, activation: bool = False):
        super(LowRankAdapter, self).__init__()
        self.device = device
        self.weight = nn.Parameter(weight.to(device))
        self.bias = nn.Parameter(torch.zeros((weight.shape[0],), device=device))
        self.act = nn.LeakyReLU(negative_slope=0.01) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        effective_weight = self.weight * self.sparsity_mask if hasattr(self, 'sparsity_mask') else self.weight

        out = F.linear(x, effective_weight, self.bias)
        out = self.act(out)
        return torch.cat([x, out], dim=-1)
