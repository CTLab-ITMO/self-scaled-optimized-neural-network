"""Дескрипторы кандидатов A: структурные и зависящие от данных.

Считаются ДО обучения (шаг 7 плана). Это принципиально: гипотезы H2 и H3
должны быть предсказаниями, а не подгонкой под уже увиденные метрики.

Деление на две группы - смысловое, а не техническое:

  структурные (n_edges, плотность, кластеризация, диаметр, спектр)
      инвариантны относительно перенумерации вершин. Значит у grid4 и
      grid4_misaligned они совпадают ПОБИТОВО - и если качество у этих двух
      графов разное, никакой структурный дескриптор его предсказать не может.

  сигнальные (энергия Дирихле, ассортативность)
      зависят и от A, и от данных, поэтому перенумерацию различают. Это
      единственные кандидаты в предикторы m(A).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


def structural_stats(A: np.ndarray) -> dict:
    d = A.shape[0]
    Af = A.astype(np.float64)
    deg = Af.sum(axis=1)
    n_edges = int(Af.sum() // 2)

    out: dict[str, float] = {
        "stat_n_edges": n_edges,
        "stat_mean_degree": float(deg.mean()),
        "stat_std_degree": float(deg.std()),
        "stat_max_degree": float(deg.max()),
        "stat_density": float(2 * n_edges / (d * (d - 1))),
        "stat_n_isolated": int((deg == 0).sum()),
    }

    n_comp, labels = connected_components(csr_matrix(A), directed=False)
    sizes = np.bincount(labels)
    giant = int(sizes.max())
    out["stat_n_components"] = int(n_comp)
    out["stat_giant_component"] = giant

    # Кластеризация через diag(A^3): треугольников у вершины = A^3[v,v] / 2.
    # networkx на плотных графах перебирает пары соседей и умирает на
    # complete (784 вершины, 783^2 пар на каждую); матмул - 1 GFLOP.
    tri = np.diag(Af @ Af @ Af) / 2.0
    pairs = deg * (deg - 1) / 2.0
    local = np.divide(tri, pairs, out=np.zeros(d), where=pairs > 0)
    out["stat_clustering"] = float(local.mean())

    # Расстояния считаем на гигантской компоненте: между компонентами путь
    # бесконечен, и среднее по всей матрице было бы просто inf.
    giant_nodes = np.flatnonzero(labels == int(np.argmax(sizes)))
    if giant_nodes.size > 1:
        sub = csr_matrix(A[np.ix_(giant_nodes, giant_nodes)])
        dist = shortest_path(sub, method="D", unweighted=True, directed=False)
        finite = dist[np.isfinite(dist)]
        off = finite[finite > 0]
        out["stat_avg_path"] = float(off.mean()) if off.size else 0.0
        out["stat_diameter"] = float(off.max()) if off.size else 0.0
    else:
        out["stat_avg_path"] = float("nan")
        out["stat_diameter"] = float("nan")

    # Спектр нормализованного лапласиана L = I - D^(-1/2) A D^(-1/2).
    # Для изолированных вершин D^(-1/2) := 0 - стандартное соглашение.
    dinv = np.zeros(d)
    nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    L = np.eye(d) - dinv[:, None] * Af * dinv[None, :]
    ev = np.linalg.eigvalsh((L + L.T) / 2.0)
    out["stat_lambda2"] = float(ev[1]) if d > 1 else float("nan")
    out["stat_lambda_max"] = float(ev[-1])
    return out


def signal_stats(A: np.ndarray, x: np.ndarray) -> dict:
    """Дескрипторы, зависящие от данных. x: [n, d], обучающая выборка.

    Энергия Дирихле считается через квадратичную форму лапласиана
    (sum по рёбрам (x_u - x_v)^2 = x^T L x), то есть одним матмулом.
    Нормировка на |E| и дисперсию делает её сравнимой между плотностями.
    """
    Af = A.astype(np.float64)
    deg = Af.sum(axis=1)
    n_edges = Af.sum() / 2.0
    if n_edges == 0:
        return {"sig_dirichlet": float("nan"), "sig_assortativity": float("nan"),
                "sig_edge_absdiff": float("nan")}

    x = x.astype(np.float64)
    n = x.shape[0]

    quad = np.einsum("ij,ij->i", x * deg, x) - np.einsum("ij,ij->i", x @ Af, x)
    var = x.var(axis=1)
    ok = var > 0
    out = {"sig_dirichlet": float((quad[ok] / n_edges / var[ok]).mean())}

    # Ассортативность сигнала: корреляция значений на концах ребра, пулом по
    # всем рёбрам и всем объектам. Считается из тех же сумм, без разворачивания
    # списка рёбер.
    two_m_n = 2.0 * n_edges * n
    s1 = float((x * deg).sum()) / two_m_n                     # E[x_u]
    s2 = float((x * x * deg).sum()) / two_m_n                 # E[x_u^2]
    sxy = float(np.einsum("ij,ij->", x @ Af, x)) / two_m_n    # E[x_u x_v]
    denom = s2 - s1 * s1
    out["sig_assortativity"] = float((sxy - s1 * s1) / denom) if denom > 0 else float("nan")

    # То же самое в единицах самих значений - удобнее читать в таблице
    out["sig_edge_absdiff"] = float(np.sqrt((quad / n_edges).mean()))
    return out


def all_stats(A: np.ndarray, x: np.ndarray) -> dict:
    return {**structural_stats(A), **signal_stats(A, x)}


if __name__ == "__main__":
    import argparse
    import time

    import pandas as pd

    from config import artifacts_dir, load_config
    from data import load_data
    from graphs import available, build_graph

    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--subsample", type=int, default=10000,
                    help="объектов для сигнальных дескрипторов")
    args = ap.parse_args()

    cfg = load_config(dev=args.dev)
    data = load_data(cfg)
    x = data.x_train[:args.subsample]

    rows = []
    for name in available():
        t0 = time.time()
        g = build_graph(name, cfg, data)
        row = {"graph": name, "group": g.group, "honest": g.honest,
               **all_stats(g.A, x)}
        row["stats_time_s"] = round(time.time() - t0, 2)
        rows.append(row)
        print(f"  {name:<17} {row['stats_time_s']:>6.2f} с", flush=True)

    df = pd.DataFrame(rows)
    out = artifacts_dir(cfg) / f"graph_stats_d{cfg.data.d}.csv"
    df.to_csv(out, index=False)

    show = ["graph", "group", "stat_n_edges", "stat_density", "stat_clustering",
            "stat_avg_path", "stat_lambda2", "sig_dirichlet", "sig_assortativity"]
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n" + df[show].to_string(index=False,
                                    float_format=lambda v: f"{v:.4f}"))
    print(f"\nсохранено: {out}")

    # Ключевая проверка: изоморфные графы обязаны совпасть по ВСЕМ структурным
    # дескрипторам и разойтись по сигнальным. Это и есть суть гипотезы H1.
    a = df[df.graph == "grid4"].iloc[0]
    b = df[df.graph == "grid4_misaligned"].iloc[0]
    struct = [c for c in df.columns if c.startswith("stat_")]
    bad = [c for c in struct if not np.isclose(a[c], b[c], rtol=1e-6, atol=1e-9)]
    print("\ngrid4 vs grid4_misaligned:")
    print(f"  структурные дескрипторы расходятся в: {bad or 'ни в одном'}")
    print(f"  энергия Дирихле:   {a.sig_dirichlet:.4f}  против  {b.sig_dirichlet:.4f}")
    print(f"  ассортативность:   {a.sig_assortativity:.4f}  против  "
          f"{b.sig_assortativity:.4f}")
    assert not bad, "изоморфные графы разошлись по структуре - это баг"
    assert not np.isclose(a.sig_dirichlet, b.sig_dirichlet, rtol=1e-3), \
        "сигнальные дескрипторы не различили согласованный и случайный граф"
    print("\n  -> структура их не различает, данные различают. H1 имеет смысл.")
