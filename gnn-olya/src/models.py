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
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Блок 1: 28x28 -> 14x14
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Блок 2: 14x14 -> 7x7
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Вход:  (B, 1, 28, 28)
        Выход: (B, 10) - логиты классов
        """
        features = self.features(x)
        logits = self.classifier(features)
        return logits


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
        return out.permute(0, 2, 3, 1).reshape(
            x.size(0),
            x.size(2) * x.size(3),
            self.dir_linears["self"].out_features
        )


class SimpleGNN(nn.Module):
    """
    GNN на 8-связной сетке:
    - Входной признак узла: только яркость (1 канал), без (x, y) координат
    - 2 слоя DirectionalConv
    - Readout: Flatten фиксированного порядка узлов -> Linear классификатор
    """
    def __init__(self, num_classes: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Слой 1: 1 -> hidden_dim
        self.conv1 = DirectionalConv(1, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Слой 2: hidden_dim -> hidden_dim
        self.conv2 = DirectionalConv(hidden_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Классификатор: 784 узла * hidden_dim -> 128 -> 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Принимает вход любого вида: (B, 1, 28, 28) или (B, 784, 1), возвращает логиты (B, 10)"""
        # Если пришло (B, 784, 1), возвращаем к (B, 1, 28, 28)
        if x.dim() == 3:
            x = x.view(x.size(0), 1, 28, 28)

        h1 = self.conv1(x)                                  # (B, 784, hidden_dim)
        h1 = self.bn1(h1.transpose(1, 2)).transpose(1, 2)
        h1 = self.relu(h1)

        B, N, C = h1.shape
        h1_2d = h1.view(B, 28, 28, C).permute(0, 3, 1, 2)  # (B, hidden_dim, 28, 28)
        h1_2d = self.pool(h1_2d)

        h2 = self.conv2(h1_2d)                              # (B, 196, 64)
        h2 = self.bn2(h2.transpose(1, 2)).transpose(1, 2)
        h2 = self.relu(h2)

        B, N, C = h2.shape
        h2_2d = h2.view(B, 14, 14, C).permute(0, 3, 1, 2)
        h2_2d = self.pool(h2_2d)

        logits = self.classifier(h2_2d)
        return logits

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
    """
    GNN для работы с любой произвольной матрицей смежности A:
    - Вход: (B, 1, 28, 28) или (B, 784, 1)
    - Permutation-invariant readout (Mean + Max pooling)
    - Точность напрямую отражает полезность матрицы A
    """
    def __init__(
        self,
        adj: torch.Tensor,
        num_classes: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3
    ):
        super().__init__()

        # Dense-версию оставляем в state_dict для совместимости со старыми checkpoint
        adj_norm = normalize_adjacency(adj)
        self.register_buffer("adj_norm", adj_norm)

        # Для вычислений используем sparse-версию
        # persistent=False: не меняет формат checkpoint/state_dict
        self.register_buffer(
            "adj_norm_sparse",
            adj_norm.to_sparse_coo().coalesce(),
            persistent=False
        )

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Первый слой: 1 -> hidden_dim
        self.layers.append(GenericGCNLayer(1, hidden_dim))
        self.bns.append(nn.LayerNorm(hidden_dim))

        # Последующие скрытые слои
        for _ in range(num_layers - 1):
            self.layers.append(GenericGCNLayer(hidden_dim, hidden_dim))
            self.bns.append(nn.LayerNorm(hidden_dim))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

        # Классификатор: Mean + Max pooling дают 2 * hidden_dim фичей
        self.classifier = nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.ReLU(),
            self.dropout,
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.view(x.size(0), 784, 1)

        h = x
        for i in range(self.num_layers):
            h_next = self.layers[i](h, self.adj_norm_sparse)
            # h_next = self.bns[i](h_next.transpose(1, 2)).transpose(1, 2)
            h_next = self.bns[i](h_next)
            if i > 0:
                h_next = h_next + h
            h = self.relu(h_next)

        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1)[0]
        graph_repr = torch.cat([mean_pool, max_pool], dim=1)

        return self.classifier(graph_repr)
