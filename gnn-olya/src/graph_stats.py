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

    tri = np.diag(Af @ Af @ Af) / 2.0
    pairs = deg * (deg - 1) / 2.0
    local = np.divide(tri, pairs, out=np.zeros(d), where=pairs > 0)
    out["stat_clustering"] = float(local.mean())

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

    dinv = np.zeros(d)
    nz = deg > 0
    dinv[nz] = deg[nz] ** -0.5
    L = np.eye(d) - dinv[:, None] * Af * dinv[None, :]
    ev = np.linalg.eigvalsh((L + L.T) / 2.0)
    out["stat_lambda2"] = float(ev[1]) if d > 1 else float("nan")
    out["stat_lambda_max"] = float(ev[-1])
    return out


if __name__ == "__main__":
    import argparse
    import time

    import pandas as pd

    from config import artifacts_dir, load_config
    from data import load_data
    from graphs import available, build_graph

    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true")
    args = ap.parse_args()

    cfg = load_config(dev=args.dev)
    data = load_data(cfg)

    rows = []
    for name in available():
        t0 = time.time()
        g = build_graph(name, cfg, data)
        row = {"graph": name, "group": g.group, "honest": g.honest,
               **structural_stats(g.A)}
        row["stats_time_s"] = round(time.time() - t0, 2)
        rows.append(row)
        print(f"  {name:<17} {row['stats_time_s']:>6.2f} с", flush=True)

    df = pd.DataFrame(rows)
    out = artifacts_dir(cfg) / f"graph_stats_d{cfg.data.d}.csv"
    df.to_csv(out, index=False)

    show = ["graph", "group", "stat_n_edges", "stat_density", "stat_clustering",
            "stat_avg_path", "stat_lambda2"]
    pd.set_option("display.width", 200, "display.max_columns", 50)
    print("\n" + df[show].to_string(index=False,
                                    float_format=lambda v: f"{v:.4f}"))
    print(f"\nсохранено: {out}")

    # Изоморфные графы обязаны совпадать по всем структурным инвариантам.
    a = df[df.graph == "grid4"].iloc[0]
    b = df[df.graph == "grid4_misaligned"].iloc[0]
    struct = [c for c in df.columns if c.startswith("stat_")]
    bad = [c for c in struct if not np.isclose(a[c], b[c], rtol=1e-6, atol=1e-9)]
    print("\ngrid4 vs grid4_misaligned:")
    print(f"  структурные дескрипторы расходятся в: {bad or 'ни в одном'}")
    assert not bad, "изоморфные графы разошлись по структуре - это баг"
