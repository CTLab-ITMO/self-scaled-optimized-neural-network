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
    """Слой графовой свёртки с 9 отдельными весами для каждого направления."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.dir_linears = nn.ModuleDict({
            dir_name: nn.Linear(in_channels, out_channels, bias=False)
            for dir_name in DIRECTIONS.keys()
        })
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape

        # Делаем паддинг нулями по краям: (B, C, 30, 30)
        padded = F.pad(x, (1, 1, 1, 1), mode="constant", value=0.0)

        out = 0.0
        # Собираем сообщения со всех 9 направлений
        for dir_name, (dr, dc) in DIRECTIONS.items():
            r_start, r_end = 1 + dr, 1 + dr + H
            c_start, c_end = 1 + dc, 1 + dc + W
            
            # neighbor_slice: (B, C, 28, 28) -> переводим в узлы (B, 784, C)
            neighbor = padded[:, :, r_start:r_end, c_start:c_end]
            neighbor_nodes = neighbor.permute(0, 2, 3, 1).reshape(B, H * W, C)
            
            out = out + self.dir_linears[dir_name](neighbor_nodes)

        return out + self.bias


class SimpleGNN(nn.Module):
    """
    GNN на 8-связной сетке.
    - Входной признак узла: только яркость (1 канал), без (x, y) координат.
    - 2 слоя DirectionalConv.
    - Readout: Flatten фиксированного порядка узлов -> Linear классификатор.
    """
    def __init__(self, num_classes: int = 10, hidden_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Слой 1: 1 -> hidden_dim
        self.conv1 = DirectionalConv(1, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        # Слой 2: hidden_dim -> hidden_dim
        self.conv2 = DirectionalConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

        # Классификатор: 784 узла * hidden_dim -> 128 -> 10
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784 * hidden_dim, 128),
            nn.ReLU(),
            self.dropout,
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Принимает вход любого вида: (B, 1, 28, 28) или (B, 784, 1), возвращает логиты (B, 10)."""
        # Если пришло (B, 784, 1), возвращаем к (B, 1, 28, 28)
        if x.dim() == 3:
            x = x.view(x.size(0), 1, 28, 28)

        h1 = self.conv1(x)                                 # (B, 784, hidden_dim)
        h1 = self.bn1(h1.transpose(1, 2)).transpose(1, 2)
        h1 = self.relu(h1)

        B, N, C = h1.shape
        h1_2d = h1.view(B, 28, 28, C).permute(0, 3, 1, 2)  # (B, hidden_dim, 28, 28)

        h2 = self.conv2(h1_2d)                              # (B, 784, hidden_dim)
        h2 = self.bn2(h2.transpose(1, 2)).transpose(1, 2)
        h2 = self.relu(h2)

        logits = self.classifier(h2)
        return logits


class GenericGCNLayer(nn.Module):
    """Слой для произвольного графа: отдельный вес для себя и для соседей"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w_self = nn.Linear(in_features, out_features, bias=False)
        self.w_neigh = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_features)
        # adj_norm: (N, N)
        neigh_msg = torch.matmul(adj_norm, x)  # агрегация сообщений соседей
        out = self.w_self(x) + self.w_neigh(neigh_msg) + self.bias
        return out


class GenericGNN(nn.Module):
    """
    GNN для работы с любой произвольной матрицей смежности A.
    - Вход: (B, 1, 28, 28) или (B, 784, 1).
    - Permutation-invariant readout (Mean + Max pooling).
    - Точность напрямую отражает полезность матрицы A.
    """
    def __init__(
        self,
        adj: torch.Tensor,
        num_classes: int = 10,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        # Сохраняем нормализованную матрицу смежности как буфер
        adj_norm = normalize_adjacency(adj)
        self.register_buffer("adj_norm", adj_norm)

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Первый слой: 1 -> hidden_dim
        self.layers.append(GenericGCNLayer(1, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Последующие скрытые слои
        for _ in range(num_layers - 1):
            self.layers.append(GenericGCNLayer(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

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
            h = self.layers[i](h, self.adj_norm)
            h = self.relu(self.bns[i](h.transpose(1, 2)).transpose(1, 2))

        mean_pool = h.mean(dim=1)
        max_pool = h.max(dim=1)[0]
        graph_repr = torch.cat([mean_pool, max_pool], dim=1)

        return self.classifier(graph_repr)