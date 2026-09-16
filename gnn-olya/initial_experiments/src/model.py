# здесь описаны сами нейросети
# создаёт GCN и MLP с нужными параметрами

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DenseGCN(nn.Module):
    """H^(l+1) = ReLU( A_hat H^(l) W^(l) + b^(l) ), readout = mean по вершинам.

    Порядок (A_hat H) W против A_hat (H W) выбирается по стоимости: матрицы
    коммутируют, но на первом слое дешевле распространить до расширения.
    Смещение прибавляется ПОСЛЕ распространения, как в GCNConv.
    """

    def __init__(self, d: int, n_layers: int, hidden: int, n_classes: int,
                 in_channels: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.d = d
        if int(in_channels or 1) != 1:
            raise ValueError("в актуальном эксперименте используется только один скалярный признак")
        dims = [1] + [hidden] * n_layers
        self.lins = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1], bias=False) for i in range(n_layers)
        )
        self.biases = nn.ParameterList(
            nn.Parameter(torch.zeros(dims[i + 1])) for i in range(n_layers)
        ) if bias else None
        self.head = nn.Linear(hidden, n_classes)

    @staticmethod
    def _propagate(A_hat: Tensor, h: Tensor) -> Tensor:
        d, b, f = h.shape
        return (A_hat @ h.reshape(d, b * f)).reshape(d, b, f)

    def _encode(self, x: Tensor, A_hat: Tensor) -> Tensor:
        """L слоёв свёртки, результат в раскладке [d, B, F]."""
        h = x.t().contiguous().unsqueeze(-1)  # [B, d] -> [d, B, 1]
        for i, lin in enumerate(self.lins):
            if lin.in_features < lin.out_features:
                h = lin(self._propagate(A_hat, h))
            else:
                h = self._propagate(A_hat, lin(h))
            if self.biases is not None:
                h = h + self.biases[i]
            h = F.relu(h)
        return h

    def forward(self, x: Tensor, A_hat: Tensor) -> Tensor:
        return self.head(self._encode(x, A_hat).mean(dim=0))  # readout

    @torch.no_grad()
    def node_embeddings(self, x: Tensor, A_hat: Tensor) -> Tensor:
        """H^(L) без readout - для диагностики oversmoothing (PLAN.md п.4)."""
        return self._encode(x, A_hat).permute(1, 0, 2)  # [B, d, hidden]


def propagation_matrix(A, kind: str = "gcn"):
    """Нормализованная матрица распространения для GCN."""
    from graphs import normalize_adj
    if kind != "gcn":
        raise ValueError(f"в актуальном эксперименте оставлена только архитектура gcn, получено {kind!r}")
    return normalize_adj(A)


class MLPReference(nn.Module):
    """Референс-потолок: обычный MLP на сырых d признаках, графа не видит.

    Сравнивать его с GNN по числу параметров бессмысленно и не нужно - он тут
    не конкурент, а верхняя отметка шкалы: сколько информации в данных есть.
    """

    def __init__(self, d: int, hidden: list[int], n_classes: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, A_hat: Tensor | None = None) -> Tensor:
        return self.net(x)


def build_model(graph_name: str, cfg) -> nn.Module:
    """graph_name == 'mlp' -> референс без графа, иначе базовый GCN."""
    if graph_name == "mlp":
        return MLPReference(
            d=cfg.data.d,
            hidden=list(cfg.mlp_reference.hidden),
            n_classes=cfg.model.n_classes,
            dropout=cfg.mlp_reference.dropout,
        )
    if cfg.model.positional_encoding:
        raise ValueError(
            "positional_encoding: true - это ломает эксперимент, "
            "модель получит идентичность вершин в обход A (PLAN.md п.0.1)"
        )
    kind = cfg.model.kind
    if kind != "gcn":
        raise ValueError(f"в актуальном эксперименте оставлена только архитектура gcn, получено {kind!r}")
    return DenseGCN(
        d=cfg.data.d,
        n_layers=cfg.model.layers,
        hidden=cfg.model.hidden,
        n_classes=cfg.model.n_classes,
        in_channels=int(getattr(cfg.model, "in_channels", 1) or 1),
        bias=cfg.model.bias,
    )


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys

    import numpy as np

    from config import load_config
    from graphs import build_graph, normalize_adj
    from data import load_data

    cfg = load_config(dev="--dev" in sys.argv)
    data = load_data(cfg)
    d = cfg.data.d
    torch.manual_seed(0)

    x = torch.from_numpy(data.x_train[:8])
    A_hat = torch.from_numpy(normalize_adj(build_graph("grid4", cfg, data).A))

    gcn = build_model("grid4", cfg)
    mlp = build_model("mlp", cfg)
    print(f"d = {d}")
    print(f"DenseGCN:     {n_params(gcn):>8} параметров")
    print(f"MLPReference: {n_params(mlp):>8} параметров")
    print(f"выход GCN: {tuple(gcn(x, A_hat).shape)}  MLP: {tuple(mlp(x).shape)}")

    # 1. Порядок (A_hat H) W и A_hat (H W) обязан давать одно и то же -
    #    иначе оптимизация по стоимости меняла бы модель.
    lin = gcn.lins[1]
    h = torch.randn(d, 4, cfg.model.hidden)
    a = DenseGCN._propagate(A_hat, lin(h))
    b = lin(DenseGCN._propagate(A_hat, h))
    print(f"\nкоммутация A_hat и W: max|разница| = {(a - b).abs().max():.2e}")
    assert torch.allclose(a, b, atol=1e-4)

    # 2. Перестановочная эквивариантность: переставим И вершины, И граф -
    #    предсказание не должно измениться. Это ядро постановки задачи.
    perm = torch.from_numpy(np.random.default_rng(0).permutation(d))
    out_ref = gcn(x, A_hat)
    out_perm = gcn(x[:, perm], A_hat[perm][:, perm])
    print(f"эквивариантность: max|разница| = {(out_ref - out_perm).abs().max():.2e}")
    assert torch.allclose(out_ref, out_perm, atol=1e-4)

    # 3. complete: после первого слоя все вершины одинаковы -> предсказание
    #    зависит только от среднего значения пикселя. Проверяем прямо: два
    #    разных объекта с одинаковым средним должны дать один ответ.
    A_full = torch.from_numpy(normalize_adj(build_graph("complete", cfg, data).A))
    emb = gcn.node_embeddings(x, A_full)
    spread = (emb - emb.mean(dim=1, keepdim=True)).abs().max()
    print(f"complete: разброс эмбеддингов по вершинам = {spread:.2e}")
    assert spread < 1e-4, "на полном графе вершины обязаны схлопнуться"

    # 4. empty: A_hat = I, вершины не общаются - выход зависит только от
    #    мультимножества значений. Перестановка ТОЛЬКО вершин ничего не меняет.
    A_eye = torch.from_numpy(normalize_adj(build_graph("empty", cfg, data).A))
    o1 = gcn(x, A_eye)
    o2 = gcn(x[:, perm], A_eye)
    print(f"empty: max|разница| при перестановке вершин = {(o1 - o2).abs().max():.2e}")
    assert torch.allclose(o1, o2, atol=1e-4)

    print("\nвсе проверки модели пройдены")
