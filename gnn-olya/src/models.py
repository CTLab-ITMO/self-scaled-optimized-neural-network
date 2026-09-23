"""
Архитектуры моделей:
1. SimpleCNN - сверточный бейзлайн
2. SimpleGNN - графовый бейзлайн на 8-связной сетке
3. GenericGNN - граф для работы с любой матрицей смежности
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.graph import DIRECTIONS
from src.graph import normalize_adjacency


class SimpleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        hidden_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.relu = nn.ReLU()

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1))

        for _ in range(num_layers - 1):
            self.convs.append(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
            )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        h = x
        for conv in self.convs:
            h = self.relu(conv(h))

        # (B, hidden_dim, 28, 28) -> (B, hidden_dim)
        graph_repr = h.mean(dim=(-2, -1))

        # Выход: (B, num_classes)
        return self.classifier(graph_repr)


class DirectionalConv(nn.Module):
    """Слой графовой свёртки с 9 отдельными весами для каждого направления"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dir_linears = nn.ModuleDict({
            dir_name: nn.Linear(in_channels, out_channels, bias=False)
            for dir_name in DIRECTIONS.keys()
        })
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.stack([
            self.dir_linears["up_left"].weight,
            self.dir_linears["up"].weight,
            self.dir_linears["up_right"].weight,
            self.dir_linears["left"].weight,
            self.dir_linears["self"].weight,
            self.dir_linears["right"].weight,
            self.dir_linears["down_left"].weight,
            self.dir_linears["down"].weight,
            self.dir_linears["down_right"].weight,
        ], dim=-1)

        kernel = weights.reshape(
            self.dir_linears["self"].out_features,
            self.dir_linears["self"].in_features,
            3,
            3
        )

        out = F.conv2d(x, kernel, bias=self.bias, padding=1)
        # Возвращает узлы: (B, 784, out_channels)
        return out.permute(0, 2, 3, 1).reshape(
            x.size(0),
            x.size(2) * x.size(3),
            self.dir_linears["self"].out_features
        )


class SimpleGNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        hidden_dim: int = 32,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.relu = nn.ReLU()

        self.convs = nn.ModuleList()
        self.convs.append(DirectionalConv(1, hidden_dim))

        for _ in range(num_layers - 1):
            self.convs.append(DirectionalConv(hidden_dim, hidden_dim))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Если пришло (B, 784, 1), преобразуем к 2D-сетке
        if x.dim() == 3:
            x = x.view(x.size(0), 1, 28, 28)

        B = x.size(0)
        h = self.relu(self.convs[0](x))

        for conv in self.convs[1:]:
            h_2d = h.view(B, 28, 28, -1).permute(0, 3, 1, 2)
            h = self.relu(conv(h_2d))

        # (B, 784, hidden_dim) -> (B, hidden_dim)
        graph_repr = h.mean(dim=1)

        # Выход: (B, num_classes)
        return self.classifier(graph_repr)


def _sparse_batch_mm(adj_sparse: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Умножает одну sparse adjacency (N, N) на batch признаков (B, N, C) без построения плотной матрицы для каждого объекта"""
    B, N, C = x.shape
    x_flat = x.permute(1, 0, 2).reshape(N, B * C)
    out_flat = torch.sparse.mm(adj_sparse, x_flat)
    return out_flat.reshape(N, B, C).permute(1, 0, 2)


class GenericGCNLayer(nn.Module):
    """Слой для произвольного графа: отдельный вес для себя и для соседей"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w_self = nn.Linear(in_features, out_features, bias=False)
        self.w_neigh = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, adj_norm_sparse: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_features)
        # adj_norm_sparse: sparse (N, N)
        neigh_msg = _sparse_batch_mm(adj_norm_sparse, x)
        out = self.w_self(x) + self.w_neigh(neigh_msg) + self.bias
        return out


class GenericGNN(nn.Module):
    def __init__(self, adj: torch.Tensor, num_classes: int = 10, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        adj_norm = normalize_adjacency(adj)
        self.register_buffer("adj_norm_sparse", adj_norm.to_sparse_coo().coalesce(), persistent=False)
        
        self.convs = nn.ModuleList()
        self.convs.append(GenericGCNLayer(1, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GenericGCNLayer(hidden_dim, hidden_dim))
            
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.view(x.size(0), 784, 1)
            
        h = x
        for conv in self.convs:
            h = self.relu(conv(h, self.adj_norm_sparse))
            
        graph_repr = h.mean(dim=1)
        return self.classifier(graph_repr)