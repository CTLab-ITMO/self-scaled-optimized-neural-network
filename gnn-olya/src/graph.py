"""
Модуль для построения графовых структур:
- get_grid8_neighbor_indices: строит карту 8 направлений для сетки HxW (MNIST)
"""

import torch

DIRECTIONS = {
    "self":        (0, 0),
    "up":          (-1, 0),
    "down":        (1, 0),
    "left":        (0, -1),
    "right":       (0, 1),
    "up_left":     (-1, -1),
    "up_right":    (-1, 1),
    "down_left":   (1, -1),
    "down_right":  (1, 1),
}


def get_grid8_neighbor_indices(height: int = 28, width: int = 28) -> dict[str, torch.Tensor]:
    """
    Для каждого направления возвращает тензор индексов соседей размера (N,).
    Если сосед выходит за границы картинки, ставится индекс N (фиктивный узел-заглушка с нулями).
    """
    num_nodes = height * width
    pad_idx = num_nodes  # индекс 784 - это padding узел

    neighbor_indices = {}

    for dir_name, (dr, dc) in DIRECTIONS.items():
        indices = torch.full((num_nodes,), pad_idx, dtype=torch.long)

        for r in range(height):
            for c in range(width):
                node_idx = r * width + c
                nr, nc = r + dr, c + dc

                # Проверяем, не вышли ли за границы изображения
                if 0 <= nr < height and 0 <= nc < width:
                    indices[node_idx] = nr * width + nc

        neighbor_indices[dir_name] = indices

    return neighbor_indices


def get_empty_adjacency(num_nodes: int = 784) -> torch.Tensor:
    """Пустой граф (вообще нет рёбер)"""
    return torch.zeros((num_nodes, num_nodes), dtype=torch.float32)


def get_grid8_adjacency(height: int = 28, width: int = 28) -> torch.Tensor:
    """Ненаправленная бинарная матрица смежности grid-8 размера (784, 784)"""
    num_nodes = height * width
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for r in range(height):
        for c in range(width):
            u = r * width + c
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        v = nr * width + nc
                        adj[u, v] = 1.0
    return adj


def get_random_adjacency(num_nodes: int = 784, num_edges: int = 2800, seed: int = 42) -> torch.Tensor:
    """Случайный симметричный граф (Erdos-Renyi) с заданным числом рёбер"""
    gen = torch.Generator().manual_seed(seed)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    total_possible = triu_indices.size(1)

    perm = torch.randperm(total_possible, generator=gen)[:num_edges]
    selected = triu_indices[:, perm]

    adj[selected[0], selected[1]] = 1.0
    adj[selected[1], selected[0]] = 1.0
    return adj


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """Симметричная GCN-нормализация: D^(-1/2) * (A + I) * D^(-1/2)"""
    deg = adj.sum(dim=1, keepdim=True)
    deg_clamped = torch.clamp(deg, min=1.0)
    adj_norm = adj / deg_clamped
    adj_norm = torch.where(deg > 0, adj_norm, torch.zeros_like(adj_norm))
    return adj_norm


if __name__ == "__main__":
    neighbors = get_grid8_neighbor_indices(28, 28)
    for name, tens in neighbors.items():
        valid = (tens != 784).sum().item()
        print(f"  {name:11s} -> валидных связей: {valid}/{784}")
