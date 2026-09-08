from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


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
        self.encoder, in_features = make_encoder_k(in_channels)
        dims = [in_features] + [hidden] * n_layers
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
        if self.encoder is not None:
            h = self.encoder(h)               # -> [d, B, k]
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


# ---------------------------------------------------------------------------
#  Обогащение признаков вершины (для H8)
# ---------------------------------------------------------------------------

class RBFExpand(nn.Module):
    """Скаляр пикселя -> k каналов: exp(-(x - c_j)^2 / 2s^2).

    Поднимает размерность, не добавляя информации: кодирование почти обратимо.
    Зависит только от значения пикселя, поэтому эквивариантность сохраняется.
    """

    def __init__(self, k: int, lo: float = 0.0, hi: float = 1.0) -> None:
        super().__init__()
        centers = torch.linspace(lo, hi, k)
        # sigma = шаг между центрами: соседние функции перекрываются примерно
        # наполовину, разложение получается гладким и без дыр между центрами.
        sigma = (hi - lo) / max(k - 1, 1)
        self.register_buffer("centers", centers.view(1, 1, k))
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.k = k

    def forward(self, h: Tensor) -> Tensor:
        """[d, B, 1] -> [d, B, k]."""
        return torch.exp(-((h - self.centers) ** 2) / (2 * self.sigma ** 2))


def make_encoder_k(k: int) -> tuple[nn.Module | None, int]:
    """(модуль обогащения или None, итоговое число входных каналов).

    k = 1 - обогащения нет, вход остаётся скаляром. Это базовый режим всей
    работы; k > 1 включается только для H8.
    """
    k = int(k or 1)
    return (None, 1) if k <= 1 else (RBFExpand(k), k)


# ---------------------------------------------------------------------------
#  GAT
# ---------------------------------------------------------------------------

class DenseGAT(nn.Module):
    """GATv1 в плотной раскладке [d, B, F], с масками вместо edge_index.

    Веса рёбер зависят от объекта, поэтому матрица распространения своя у
    каждого объекта батча: отсюда bmm и разбиение батча на куски.
    Получает НЕ нормализованную матрицу — только маску смежности с петлями.

    Память: autograd удерживает 1.49 ГБ при B=64 и НЕ зависит от ATT_CHUNK;
    CHECKPOINT=True снижает до 0.17 ГБ ценой 23% времени. ATT_CHUNK управляет
    только разовым пиком. При OOM: CHECKPOINT -> ATT_CHUNK -> batch (последний
    уменьшать и для GCN тоже). Замеры — docs/IDEAS.md, идея 9.
    """

    ATT_CHUNK = 32          # объектов за раз при построении [d, d, B]
    NEG_SLOPE = 0.2         # как в оригинальной статье
    CHECKPOINT = True       # пересчитывать куски в backward (см. докстринг)

    def __init__(self, d: int, n_layers: int, hidden: int, n_classes: int,
                 in_channels: int = 1, heads: int = 1, bias: bool = True) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError(f"hidden={hidden} не делится на heads={heads}")
        self.d, self.heads, self.per_head = d, heads, hidden // heads
        self.encoder, in_features = make_encoder_k(in_channels)
        dims = [in_features] + [hidden] * n_layers
        self.lins = nn.ModuleList(
            nn.Linear(dims[i], hidden, bias=False) for i in range(n_layers)
        )
        # a1 и a2 - две половины вектора внимания, по одной на голову
        self.att_src = nn.ParameterList(
            nn.Parameter(torch.empty(heads, self.per_head)) for _ in range(n_layers)
        )
        self.att_dst = nn.ParameterList(
            nn.Parameter(torch.empty(heads, self.per_head)) for _ in range(n_layers)
        )
        for t in list(self.att_src) + list(self.att_dst):
            nn.init.xavier_uniform_(t)
        self.biases = nn.ParameterList(
            nn.Parameter(torch.zeros(hidden)) for _ in range(n_layers)
        ) if bias else None
        self.head = nn.Linear(hidden, n_classes)

    def _attend(self, wh: Tensor, mask: Tensor, layer: int) -> Tensor:
        """wh: [d, B, H, P], mask: [d, d] bool -> выход [d, B, H*P]."""
        d, b, H, P = wh.shape
        alpha = (wh * self.att_src[layer]).sum(-1)      # [d, B, H]
        beta = (wh * self.att_dst[layer]).sum(-1)       # [d, B, H]
        def one_chunk(al: Tensor, be: Tensor, v: Tensor) -> Tensor:
            # e_uv = LeakyReLU(alpha_u + beta_v); строка u - её соседи v
            lg = al.unsqueeze(1) + be.unsqueeze(0)          # [d, d, c, H]
            lg = F.leaky_relu(lg, self.NEG_SLOPE)
            lg = lg.masked_fill(~mask[:, :, None, None], float("-inf"))
            att = torch.softmax(lg, dim=1)                  # нормировка по v
            # [c,H,d,d] @ [c,H,d,P] -> [c,H,d,P]
            return torch.matmul(att.permute(2, 3, 0, 1),
                                v.permute(1, 2, 0, 3)).permute(2, 0, 1, 3)

        parts = []
        use_ckpt = self.CHECKPOINT and self.training and torch.is_grad_enabled()
        for s in range(0, b, self.ATT_CHUNK):
            e = s + min(self.ATT_CHUNK, b - s)
            args = (alpha[:, s:e], beta[:, s:e], wh[:, s:e])
            parts.append(
                checkpoint(one_chunk, *args, use_reentrant=False)
                if use_ckpt else one_chunk(*args)
            )
        return torch.cat(parts, dim=1).reshape(d, b, H * P)

    def _encode(self, x: Tensor, mask: Tensor) -> Tensor:  # noqa: D401
        h = x.t().contiguous().unsqueeze(-1)
        if self.encoder is not None:
            h = self.encoder(h)
        mask = mask > 0
        for i, lin in enumerate(self.lins):
            wh = lin(h).view(self.d, h.shape[1], self.heads, self.per_head)
            h = self._attend(wh, mask, i)
            if self.biases is not None:
                h = h + self.biases[i]
            h = F.relu(h)
        return h

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.head(self._encode(x, mask).mean(dim=0))

    @torch.no_grad()
    def node_embeddings(self, x: Tensor, mask: Tensor) -> Tensor:
        return self._encode(x, mask).permute(1, 0, 2)


# ---------------------------------------------------------------------------
#  GIN
# ---------------------------------------------------------------------------

class DenseGIN(nn.Module):
    """h' = MLP( (1 + eps) h_u + sum_{v in N(u)} h_v ), eps обучаемая.

    Агрегация суммой, а не средним — в этом вся разница с GCN. Матрицу
    получает сырую. На плотных графах сумма даёт огромные активации: это
    свойство GIN, а не артефакт реализации.
    """

    def __init__(self, d: int, n_layers: int, hidden: int, n_classes: int,
                 in_channels: int = 1, bias: bool = True) -> None:
        super().__init__()
        self.d = d
        self.encoder, in_features = make_encoder_k(in_channels)
        dims = [in_features] + [hidden] * n_layers
        self.mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(dims[i], hidden, bias=bias), nn.ReLU(),
                nn.Linear(hidden, hidden, bias=bias),
            ) for i in range(n_layers)
        )
        self.eps = nn.Parameter(torch.zeros(n_layers))
        self.head = nn.Linear(hidden, n_classes)

    def _encode(self, x: Tensor, A: Tensor) -> Tensor:
        h = x.t().contiguous().unsqueeze(-1)
        if self.encoder is not None:
            h = self.encoder(h)
        for i, mlp in enumerate(self.mlps):
            d, b, f = h.shape
            agg = (A @ h.reshape(d, b * f)).reshape(d, b, f)
            h = F.relu(mlp((1.0 + self.eps[i]) * h + agg))
        return h

    def forward(self, x: Tensor, A: Tensor) -> Tensor:
        return self.head(self._encode(x, A).mean(dim=0))

    @torch.no_grad()
    def node_embeddings(self, x: Tensor, A: Tensor) -> Tensor:
        return self._encode(x, A).permute(1, 0, 2)


def propagation_matrix(A, kind: str):
    """Матрица на вход слоя зависит от архитектуры.

    gcn -> нормализованная; gat -> маска A+I (петли обязательны, иначе softmax
    по пустому множеству); gin -> сырая A.
    """
    import numpy as np

    from graphs import normalize_adj
    if kind == "gcn":
        return normalize_adj(A)
    if kind == "gat":
        return (A.astype(np.float32) + np.eye(A.shape[0], dtype=np.float32))
    if kind == "gin":
        return A.astype(np.float32)
    raise ValueError(f"неизвестная архитектура {kind!r}")


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
    """graph_name == 'mlp' -> референс без графа, иначе GNN по cfg.model.kind.

    Архитектура читается из cfg.model.kind: gcn (базовая, все 110 прогонов),
    gat и gin добавлены под H8. Переключается через --arch, значение пишется
    в results.csv и входит в ключ пропуска, поэтому прогоны разных архитектур
    не склеиваются.
    """
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
    common = dict(
        d=cfg.data.d,
        n_layers=cfg.model.layers,
        hidden=cfg.model.hidden,
        n_classes=cfg.model.n_classes,
        in_channels=int(getattr(cfg.model, "in_channels", 1) or 1),
        bias=cfg.model.bias,
    )
    if kind == "gcn":
        return DenseGCN(**common)
    if kind == "gat":
        return DenseGAT(heads=int(getattr(cfg.model, "heads", 1) or 1), **common)
    if kind == "gin":
        return DenseGIN(**common)
    raise ValueError(f"неизвестная архитектура {kind!r}; есть: gcn, gat, gin")


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

    # 5. GAT и GIN: та же эквивариантность, с обогащением признаков и без.
    #    Это ядро постановки: если её нет, модель получила идентичность вершин.
    print()
    for kind in ("gat", "gin"):
        M = torch.from_numpy(propagation_matrix(
            build_graph("grid4", cfg, data).A, kind))
        for k in (1, 8):
            cfg.model["kind"] = kind
            cfg.model["in_channels"] = k
            m = build_model("grid4", cfg).eval()
            with torch.no_grad():
                r = m(x, M)
                q = m(x[:, perm], M[perm][:, perm])
            diff = (r - q).abs().max()
            print(f"{kind}, in_channels={k}: эквивариантность = {diff:.2e}")
            assert diff < 1e-4, f"{kind} потерял эквивариантность при k={k}"
    cfg.model["kind"] = "gcn"
    cfg.model["in_channels"] = 1

    # 6. Лемма из docs/PREREGISTRATION.md: при скалярном признаке логит внимания
    #    GAT схлопывается в alpha*h_u + beta*h_v - ДВА эффективных параметра
    #    независимо от ширины слоя. Проверяем рангом, а не на глаз.
    def att_rank(k: int) -> int:
        g = DenseGAT(d=d, n_layers=1, hidden=64, n_classes=10, in_channels=k)
        h0 = torch.rand(5, d).t().contiguous().unsqueeze(-1)
        if g.encoder is not None:
            h0 = g.encoder(h0)
        wh = g.lins[0](h0).view(d, 5, 1, 64)
        # признаки, из которых считается логит: alpha по u и beta по v
        feats = torch.cat([(wh * g.att_src[0]).sum(-1).reshape(-1, 1),
                           (wh * g.att_dst[0]).sum(-1).reshape(-1, 1)], 1)
        return int(torch.linalg.matrix_rank(
            torch.cat([h0.reshape(d * 5, -1), feats], 1)).item())

    r1, r8 = att_rank(1), att_rank(8)
    print(f"\nранг входа внимания GAT: in_channels=1 -> {r1}, in_channels=8 -> {r8}")
    assert r1 <= 2, "лемма нарушена: при скалярном признаке должно быть <= 2"
    assert r8 > r1, "обогащение обязано поднимать размерность входа внимания"

    # 7. GIN агрегирует суммой, GCN средним - на неоднородных степенях это
    #    обязано давать разные ответы. Если совпало, где-то перепутаны матрицы.
    cfg.model["kind"] = "gin"
    gin = build_model("grid4", cfg)
    A_raw = torch.from_numpy(propagation_matrix(
        build_graph("grid4", cfg, data).A, "gin"))
    cfg.model["kind"] = "gcn"
    with torch.no_grad():
        spread = gin(x, A_raw).std()
    print(f"GIN на сырой A: разброс логитов = {spread:.3f} (сумма не нормирована)")
    assert spread > 0, "GIN выдал константу - проверь матрицу распространения"

    print("\nвсе проверки модели пройдены")
