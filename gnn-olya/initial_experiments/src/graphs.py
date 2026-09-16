# создаёт графы, на которых обучается GNN

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


@dataclass
class Graph:
    name: str
    group: str
    honest: bool  # False у оракулов: используют знание pi, это не метод
    A: np.ndarray  # uint8 [d, d]
    meta: dict = field(default_factory=dict)


BuilderFn = Callable[..., np.ndarray]
_BUILDERS: dict[str, tuple[BuilderFn, str, bool]] = {}


def register(name: str, group: str, honest: bool = True):
    def deco(fn: BuilderFn) -> BuilderFn:
        if name in _BUILDERS:
            raise ValueError(f"кандидат {name} уже зарегистрирован")
        _BUILDERS[name] = (fn, group, honest)
        return fn

    return deco


def available() -> list[str]:
    return list(_BUILDERS)



def validate(A: np.ndarray, d: int) -> None:
    if A.shape != (d, d):
        raise ValueError(f"форма {A.shape}, ожидалась ({d}, {d})")
    if not np.array_equal(A, A.T):
        raise ValueError("матрица не симметрична")
    if np.any(np.diag(A) != 0):
        raise ValueError("диагональ не нулевая")
    if not np.isin(A, (0, 1)).all():
        raise ValueError("матрица не бинарная")


def basic_meta(A: np.ndarray) -> dict:
    d = A.shape[0]
    deg = A.sum(axis=1)
    n_edges = int(A.sum() // 2)
    n_comp, labels = connected_components(csr_matrix(A), directed=False)
    giant = int(np.bincount(labels).max())
    return {
        "n_nodes": d,
        "n_edges": n_edges,
        "mean_degree": float(deg.mean()),
        "std_degree": float(deg.std()),
        "max_degree": int(deg.max()),
        "density": float(2 * n_edges / (d * (d - 1))) if d > 1 else 0.0,
        "n_isolated": int((deg == 0).sum()),
        "n_components": int(n_comp),
        "giant_component": giant,
    }


# Ключи basic_meta. Нужны train.py, чтобы строка референса mlp имела ТЕ ЖЕ
# поля, что и строки графов: заголовок results.csv берётся из первой строки,
# и если первой окажется более бедная, остальные колонки молча пропадут.
META_KEYS = ("n_nodes", "n_edges", "mean_degree", "std_degree", "max_degree",
             "density", "n_isolated", "n_components", "giant_component")


def normalize_adj(A: np.ndarray) -> np.ndarray:
    """A_hat = D^(-1/2) (A + I) D^(-1/2), где D - степени графа с петлями.

    Петли гарантируют deg >= 1, поэтому изолированные вершины не дают деления
    на ноль: у них A_hat[v, v] = 1, то есть вершина просто сохраняет своё
    значение. Это ровно тот вырожденный случай, который даёт `empty`.
    """
    A_loop = A.astype(np.float64) + np.eye(A.shape[0])
    dinv = 1.0 / np.sqrt(A_loop.sum(axis=1))
    return (dinv[:, None] * A_loop * dinv[None, :]).astype(np.float32)


def build_graph(name: str, cfg, data) -> Graph:
    if name not in _BUILDERS:
        raise KeyError(f"неизвестный кандидат {name!r}; есть: {available()}")
    fn, group, honest = _BUILDERS[name]
    A = np.ascontiguousarray(fn(cfg=cfg, data=data).astype(np.uint8))
    validate(A, cfg.data.d)
    return Graph(name=name, group=group, honest=honest, A=A, meta=basic_meta(A))


# --------------------------------------------------------------------------
#  вспомогательное
# --------------------------------------------------------------------------

def grid_adjacency(side: int, connectivity: int = 4) -> np.ndarray:
    """Решётка side x side в ИСХОДНЫХ координатах пикселей (до перестановки)."""
    idx = np.arange(side * side).reshape(side, side)
    A = np.zeros((side * side, side * side), dtype=np.uint8)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    elif connectivity != 4:
        raise ValueError("connectivity должно быть 4 или 8")
    for dr, dc in offsets:
        r0, r1 = max(0, -dr), side - max(0, dr)
        c0, c1 = max(0, -dc), side - max(0, dc)
        src = idx[r0:r1, c0:c1].ravel()
        dst = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc].ravel()
        A[src, dst] = 1
    return A


def relabel(A: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """P_perm A P_perm^T, то есть A[perm[u], perm[v]].

    Данные переставлены как x_tilde[v] = x[pi[v]] - вершина v держит пиксель
    pi[v]. Значит граф "соседние пиксели" в координатах вершин это
    A_grid[pi[u], pi[v]]. Ровно то же самое даёт relabel(A_grid, pi).
    """
    return A[np.ix_(perm, perm)]


# --------------------------------------------------------------------------
#  R - референсы
# --------------------------------------------------------------------------

@register("empty", group="R")
def _empty(cfg, data) -> np.ndarray:
    """Пол шкалы. A_hat = I, модель = функция гистограммы интенсивностей."""
    return np.zeros((cfg.data.d, cfg.data.d), dtype=np.uint8)


@register("complete", group="R")
def _complete(cfg, data) -> np.ndarray:
    """Патология. A_hat = J/d, после первого слоя все вершины одинаковы,
    модель = функция среднего значения пикселя."""
    d = cfg.data.d
    return (np.ones((d, d), dtype=np.uint8) - np.eye(d, dtype=np.uint8))


# --------------------------------------------------------------------------
#  O - оракулы (нечестные: используют знание pi)
# --------------------------------------------------------------------------

@register("grid4", group="O", honest=False)
def _grid4(cfg, data) -> np.ndarray:
    return relabel(grid_adjacency(cfg.data.side, 4), data.pi)


@register("grid8", group="O", honest=False)
def _grid8(cfg, data) -> np.ndarray:
    return relabel(grid_adjacency(cfg.data.side, 8), data.pi)


# --------------------------------------------------------------------------
#  D - data-driven: считаются ТОЛЬКО по train
# --------------------------------------------------------------------------

def pixel_correlation(data) -> np.ndarray:
    """|corr| между пикселями по train, кешируется на data.

    Только train: утечки нет по построению. Мёртвые пиксели (std = 0) получают
    нулевую похожесть и остаются изолированными — осознанно, а не потерянный NaN.
    """
    cached = getattr(data, "_corr_cache", None)
    if cached is not None:
        return cached

    x = data.x_train
    d = x.shape[1]
    alive = x.std(axis=0) > 0
    S = np.zeros((d, d), dtype=np.float64)
    if alive.sum() > 1:
        C = np.nan_to_num(np.corrcoef(x[:, alive], rowvar=False), nan=0.0)
        S[np.ix_(alive, alive)] = np.abs(C)
    np.fill_diagonal(S, 0.0)
    data._corr_cache = S
    return S


def knn_from_similarity(S: np.ndarray, k: int) -> np.ndarray:
    """kNN по похожести, симметризация через OR.

    Рёбра только к соседям с положительной корреляцией: иначе argsort добирает
    ничьи на нуле случайными вершинами, и мёртвые пиксели перестают быть
    изолированными.
    """
    d = S.shape[0]
    A = np.zeros((d, d), dtype=np.uint8)
    top = np.argsort(-S, axis=1)[:, :k]
    rows = np.repeat(np.arange(d), k)
    cols = top.ravel()
    keep = S[rows, cols] > 0
    A[rows[keep], cols[keep]] = 1
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return A


def top_m_edges(S: np.ndarray, m: int) -> np.ndarray:
    """Ровно m самых похожих пар - точное совпадение плотности с образцом."""
    d = S.shape[0]
    iu = np.triu_indices(d, 1)
    vals = S[iu]
    m = min(m, int((vals > 0).sum()))
    pick = np.argpartition(-vals, m - 1)[:m] if m > 0 else np.array([], int)
    A = np.zeros((d, d), dtype=np.uint8)
    A[iu[0][pick], iu[1][pick]] = 1
    return np.maximum(A, A.T)


def _register_knn_corr(k: int) -> None:
    @register(f"knn_corr_k{k}", group="D")
    def _builder(cfg, data, _k=k) -> np.ndarray:
        return knn_from_similarity(pixel_correlation(data), _k)


for _k in (4, 8, 16):
    _register_knn_corr(_k)


@register("thr_corr", group="D")
def _thr_corr(cfg, data) -> np.ndarray:
    """Порог на |corr|, подобранный под плотность grid4 - честный аналог
    оракула при том же бюджете рёбер."""
    m = int(grid_adjacency(cfg.data.side, 4).sum() // 2)
    return top_m_edges(pixel_correlation(data), m)



# --------------------------------------------------------------------------
#  C - контроли. Каждый согласован по плотности со своим оригиналом.
# --------------------------------------------------------------------------

def _rng(cfg, salt: int = 0) -> np.random.Generator:
    """Сид графов, НЕ сид прогона: граф обязан быть одинаковым во всех сидах."""
    return np.random.default_rng(cfg.graphs.graph_seed + salt)


def er_exact(d: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """Эрдёш-Реньи ровно с m рёбрами (а не с p): плотность совпадает точно,
    без случайного разброса числа рёбер."""
    iu = np.triu_indices(d, 1)
    pick = rng.choice(iu[0].size, size=m, replace=False)
    A = np.zeros((d, d), dtype=np.uint8)
    A[iu[0][pick], iu[1][pick]] = 1
    return np.maximum(A, A.T)


def _from_nx(G, d: int) -> np.ndarray:
    import networkx as nx

    A = nx.to_numpy_array(G, nodelist=range(d), dtype=np.uint8)
    A = (A > 0).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return np.maximum(A, A.T)


@register("grid4_misaligned", group="C")
def _grid4_misaligned(cfg, data) -> np.ndarray:
    """ГЛАВНЫЙ КОНТРОЛЬ: та же решётка, приклеенная случайной sigma вместо pi.

    Изоморфна grid4 — все структурные дескрипторы совпадают побитово.
    Отличается только согласованность с данными.
    """
    sigma = _rng(cfg, salt=1).permutation(cfg.data.d)
    return relabel(grid_adjacency(cfg.data.side, 4), sigma)


def _register_misaligned(k: int) -> None:
    """Ещё перестановки sigma: выборка вместо одной.

    Базовый `grid4_misaligned` использует одну sigma, поэтому вывод формально
    опирался на n = 1. Все изоморфны grid4 так же.
    """
    name = f"grid4_misaligned_s{k}"

    @register(name, group="C")
    def _builder(cfg, data, _k=k) -> np.ndarray:  # noqa: ANN001
        sigma = _rng(cfg, salt=100 + _k).permutation(cfg.data.d)
        return relabel(grid_adjacency(cfg.data.side, 4), sigma)


for _k in range(2, 11):
    _register_misaligned(_k)


@register("er_grid4", group="C")
def _er_grid4(cfg, data) -> np.ndarray:
    m = int(grid_adjacency(cfg.data.side, 4).sum() // 2)
    return er_exact(cfg.data.d, m, _rng(cfg, salt=2))


@register("er_knn8", group="C")
def _er_knn8(cfg, data) -> np.ndarray:
    m = int(knn_from_similarity(pixel_correlation(data), 8).sum() // 2)
    return er_exact(cfg.data.d, m, _rng(cfg, salt=3))


@register("regular4", group="C")
def _regular4(cfg, data) -> np.ndarray:
    import networkx as nx

    return _from_nx(
        nx.random_regular_graph(4, cfg.data.d, seed=cfg.graphs.graph_seed + 4),
        cfg.data.d,
    )


@register("config_knn8", group="C")
def _config_knn8(cfg, data) -> np.ndarray:
    """Конфигурационная модель на степенной последовательности knn_corr_k8:
    те же степени, случайная топология. Отделяет степени от структуры.

    Упрощение мультиграфа съедает часть рёбер, поэтому степени совпадают
    приблизительно, а не точно - фактическое число рёбер пишется в meta.
    """
    import networkx as nx

    A8 = knn_from_similarity(pixel_correlation(data), 8)
    seq = A8.sum(axis=1).astype(int).tolist()
    if sum(seq) % 2:
        seq[int(np.argmax(seq))] += 1
    G = nx.configuration_model(seq, seed=cfg.graphs.graph_seed + 5)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    return _from_nx(G, cfg.data.d)


# --------------------------------------------------------------------------
#  W - кривая деградации: решётка -> случайный граф с ТЕМИ ЖЕ степенями
# --------------------------------------------------------------------------

# Единственный источник истины для точек кривой. analyze.py импортирует
# отсюда же: если продублировать список в конфиге и читать его там, правка
# конфига молча разошлась бы с именами кандидатов и кривая построилась бы
# по пустому множеству точек.
WS_P = (0.05, 0.1, 0.25, 0.5, 1.0)


def ws_tag(p: float) -> str:
    return f"ws_p{p:g}".replace(".", "")


def _register_ws(p: float) -> None:
    @register(ws_tag(p), group="W")
    def _builder(cfg, data, _p=p) -> np.ndarray:
        """Двойные перестановки рёбер на доле _p рёбер.

        Не классический Уоттс-Строгатц: двойная перестановка сохраняет степени
        ТОЧНО, поэтому меняется ровно одно — локальность связей.
        При _p = 1 это |E| перестановок, а не полная рандомизация: ws_p1
        сохраняет остаточную локальность и не совпадает с regular4.
        """
        import networkx as nx

        if list(cfg.graphs.ws_p) != list(WS_P):
            raise ValueError(
                f"graphs.ws_p в конфиге {list(cfg.graphs.ws_p)} разошёлся с "
                f"WS_P в graphs.py {list(WS_P)} - кривая построится неверно"
            )
        A = grid_adjacency(cfg.data.side, 4)
        G = nx.from_numpy_array(relabel(A, data.pi))
        n_swap = int(round(_p * G.number_of_edges()))
        if n_swap > 0:
            nx.double_edge_swap(G, nswap=n_swap, max_tries=n_swap * 100,
                                seed=cfg.graphs.graph_seed + 6)
        return _from_nx(G, cfg.data.d)


for _p in WS_P:
    _register_ws(_p)


# --------------------------------------------------------------------------
#  P - патологии. Задают шкалу, в таблицу попадают наравне с остальными.
# --------------------------------------------------------------------------

@register("star", group="P")
def _star(cfg, data) -> np.ndarray:
    d = cfg.data.d
    A = np.zeros((d, d), dtype=np.uint8)
    A[0, 1:] = 1
    A[1:, 0] = 1
    return A


@register("path", group="P")
def _path(cfg, data) -> np.ndarray:
    d = cfg.data.d
    A = np.zeros((d, d), dtype=np.uint8)
    idx = np.arange(d - 1)
    A[idx, idx + 1] = 1
    return np.maximum(A, A.T)


@register("blocks", group="P")
def _blocks(cfg, data) -> np.ndarray:
    """Кластеры по корреляциям, внутри - клики, между - пусто. Проверяет,
    что даёт сильная локальная связность без дальних связей."""
    from sklearn.cluster import KMeans

    S = pixel_correlation(data)
    labels = KMeans(n_clusters=cfg.graphs.n_blocks, n_init=10,
                    random_state=cfg.graphs.graph_seed).fit_predict(S)
    A = (labels[:, None] == labels[None, :]).astype(np.uint8)
    np.fill_diagonal(A, 0)
    return A


if __name__ == "__main__":
    import sys

    from config import load_config
    from data import load_data

    cfg = load_config(dev="--dev" in sys.argv)
    data = load_data(cfg)
    print(f"d = {cfg.data.d}, side = {cfg.data.side}\n")
    header = f"{'кандидат':<17} {'гр':<3} {'честн':<6} {'|E|':>8} {'ср.степ':>8} " \
             f"{'плотн':>9} {'изолир':>7} {'компон':>7}"
    print(header)
    print("-" * len(header))
    for nm in available():
        g = build_graph(nm, cfg, data)
        m = g.meta
        print(f"{nm:<17} {g.group:<3} {str(g.honest):<6} {m['n_edges']:>8} "
              f"{m['mean_degree']:>8.2f} {m['density']:>9.5f} "
              f"{m['n_isolated']:>7} {m['n_components']:>7}")

    graphs = {nm: build_graph(nm, cfg, data) for nm in available()}

    # решётка side x side, 4-связная: 2 * side * (side - 1) рёбер
    g4 = graphs["grid4"]
    expected = 2 * cfg.data.side * (cfg.data.side - 1)
    assert g4.meta["n_edges"] == expected, (g4.meta["n_edges"], expected)
    print(f"\ngrid4: рёбер {g4.meta['n_edges']} = ожидаемые {expected}, ок")

    # перенос через pi не должен менять структуру - только нумерацию
    raw = grid_adjacency(cfg.data.side, 4)
    assert raw.sum() == g4.A.sum(), "перенос через pi изменил число рёбер"
    assert np.array_equal(np.sort(raw.sum(1)), np.sort(g4.A.sum(1))), \
        "перенос через pi изменил распределение степеней"
    print("grid4: изоморфен исходной решётке, ок")

    # ГЛАВНЫЙ КОНТРОЛЬ: изоморфен grid4, значит ВСЕ структурные
    # характеристики обязаны совпасть побитово. Если разойдутся - контроль
    # сломан, и вывод про класс изоморфизма делать будет нельзя.
    mis = graphs["grid4_misaligned"]
    # Дробные характеристики сравниваем с допуском: суммирование идёт в другом
    # порядке, и last-bit расхождение тут не признак разной структуры.
    diff = {
        k for k, v in g4.meta.items()
        if (not np.isclose(v, mis.meta[k], rtol=1e-9, atol=1e-9)
            if isinstance(v, float) else v != mis.meta[k])
    }
    assert not diff, f"grid4_misaligned отличается от grid4 по: {diff}"
    assert not np.array_equal(g4.A, mis.A), "sigma совпала с pi, контроль пустой"
    print("grid4_misaligned: структурно неотличим от grid4, но не равен ему, ок")

    # Контроли по плотности обязаны совпадать с образцом по числу рёбер
    pairs = [("er_grid4", "grid4"), ("er_knn8", "knn_corr_k8"),
             ("thr_corr", "grid4")]
    for ctrl, orig in pairs:
        a, b = graphs[ctrl].meta["n_edges"], graphs[orig].meta["n_edges"]
        assert a == b, f"{ctrl}: рёбер {a}, у образца {orig} {b}"
        print(f"{ctrl:<16} плотность совпала с {orig} ({a} рёбер), ок")

    # Кривая W: двойные перестановки обязаны сохранять И число рёбер,
    # И степенную последовательность точно
    deg4 = np.sort(g4.A.sum(1))
    for nm in [n for n in available() if n.startswith("ws_p")]:
        g = graphs[nm]
        assert g.meta["n_edges"] == g4.meta["n_edges"], f"{nm}: изменилось |E|"
        assert np.array_equal(np.sort(g.A.sum(1)), deg4), f"{nm}: изменились степени"
    print("ws_p*: |E| и степени совпадают с grid4 точно, ок")

    # Случайные графы не должны зависеть от сида прогона - только от graph_seed
    for nm in ("er_grid4", "grid4_misaligned", "regular4"):
        assert np.array_equal(build_graph(nm, cfg, data).A, graphs[nm].A), \
            f"{nm}: повторная сборка дала другой граф"
    print("случайные кандидаты воспроизводимы, ок")

    # A_hat: строки нормированы, изолированных делений на ноль нет
    for nm in ("empty", "complete", "grid4"):
        A_hat = normalize_adj(build_graph(nm, cfg, data).A)
        assert np.isfinite(A_hat).all(), f"{nm}: в A_hat есть nan/inf"
    print("A_hat: конечен для всех кандидатов, ок")

    A_hat_empty = normalize_adj(build_graph("empty", cfg, data).A)
    assert np.allclose(A_hat_empty, np.eye(cfg.data.d)), "empty: A_hat != I"
    print("empty: A_hat = I, ок (модель вырождается в MLP по вершинам + среднее)")

    A_hat_full = normalize_adj(build_graph("complete", cfg, data).A)
    assert np.allclose(A_hat_full, np.ones((cfg.data.d, cfg.data.d)) / cfg.data.d), \
        "complete: A_hat != J/d"
    print("complete: A_hat = J/d, ок (после 1 слоя все вершины одинаковы)")
