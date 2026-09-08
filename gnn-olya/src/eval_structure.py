# проверяет, насколько хорошо методы восстанавливают настоящий граф по данным
# запускает тесты на синтетических графах и сохраняет результаты в structure.csv

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import structure as ST  # noqa: E402
from config import load_config, results_dir  # noqa: E402
from synth import make_clusters, recovery  # noqa: E402

GRAPHS = ("tree", "grid", "smallworld", "community", "scalefree",
          "ring", "geometric")

METHODS = {
    "corr_static":    ("corr", "given", "корреляция, плотность задана"),
    "partial_static": ("partial", "given", "частная корр., плотность задана"),
    "partial_free":   ("auto", "stability", "оценка, сжатие и плотность выбираются вслепую"),
}

KEY = ["graph", "seed", "method", "n"]


def build(method: str, x: np.ndarray, n_true: int, seed: int):
    score, density, _ = METHODS[method]
    if score == "corr":
        return ST.top_edges(ST.corr_abs(x), n_true), float("nan"), "corr"
    if density == "given":
        f = ST.fit(x, score=score, n_edges=n_true, seed=seed)
    else:
        f = ST.fit(x, score=score, n_edges=None, seed=seed)
    return f.A, f.shrink, f.score


def oracle_f1(score: str, x: np.ndarray, A_true: np.ndarray,
              n_true: int) -> float:
    """Лучший достижимый F1 по сетке параметров, только для диагностики."""
    if score == "corr":
        return recovery(ST.top_edges(ST.corr_abs(x), n_true), A_true)["f1"]
    scores = ("partial", "partial_aracne") if score == "auto" else (score,)
    return max(
        recovery(ST.top_edges(ST.SCORES[sc](x, sh), n_true), A_true)["f1"]
        for sc in scores for sh in ST.SHRINK_GRID
    )


def n_triangles(A: np.ndarray) -> int:
    """Число треугольников; полезно для интерпретации выбора auto."""
    Ai = A.astype(np.int64)
    return int(np.trace(Ai @ Ai @ Ai) // 6)


def run(graphs, seeds, n_grid, side: int, out: Path) -> int:
    import pandas as pd

    done = set()
    if out.exists():
        old = pd.read_csv(out)
        if set(KEY).issubset(old.columns):
            done = set(map(tuple, old[KEY].values))

    total = len(graphs) * len(seeds) * len(n_grid) * len(METHODS)
    i = 0
    for g in graphs:
        for seed in seeds:
            for n in n_grid:
                task = None
                for method in METHODS:
                    i += 1
                    key = (g, seed, method, n)
                    if key in done:
                        print(f"[{i}/{total} сделан] {method} / {g} / сид {seed} / n={n}",
                              flush=True)
                        continue
                    if task is None:
                        task = make_clusters(side=side, n=n, n_classes=4,
                                             m=24, seed=seed, graph=g)
                    x = task.x_train.astype(np.float64)
                    n_true = int(task.A_true.sum() // 2)
                    t0 = time.time()
                    A, sh, sc = build(method, x, n_true, seed)
                    r = recovery(A, task.A_true)
                    row = dict(
                        graph=g, seed=seed, method=method, n=n,
                        n_edges=int(A.sum() // 2), n_true=n_true,
                        n_train=len(x), tri_true=n_triangles(task.A_true),
                        precision=round(r["precision"], 4),
                        recall=round(r["recall"], 4),
                        f1=round(r["f1"], 4), shrink=sh, score=sc,
                        oracle_f1=round(oracle_f1(METHODS[method][0], x,
                                                  task.A_true, n_true), 4),
                        sec=round(time.time() - t0, 1),
                    )
                    print(f"[{i}/{total}] {method:15s} {g:11s} сид {seed} n={n:<5d} "
                          f"рёбер {row['n_edges']:4d}/{n_true:<4d} "
                          f"f1={row['f1']:.3f} (оракул {row['oracle_f1']:.3f}) "
                          f"[{row['sec']:.0f} c]", flush=True)
                    cur = pd.DataFrame([row])
                    if out.exists():
                        cur = pd.concat([pd.read_csv(out), cur], ignore_index=True)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cur.to_csv(out, index=False)
    return 0


def summarize(out: Path) -> int:
    import pandas as pd

    if not out.exists():
        print(f"нет {out} - сначала запусти без --summary")
        return 1
    df = pd.read_csv(out)
    order = [m for m in METHODS if m in set(df.method)]
    base = df[df.n == df.n.max()]
    if base.empty:
        base = df

    ns = sorted(df.seed.unique())
    print(f"\nсидов: {len(ns)}   объектов: {sorted(df.n.unique())}")

    print("\nВОССТАНОВЛЕНИЕ истинного графа (F1), среднее по сидам")
    print("-" * 78)
    t = base.pivot_table(index="method", columns="graph", values="f1")
    cols = [g for g in GRAPHS if g in t.columns]
    print(t.reindex(index=order, columns=cols).round(3).to_string())

    if len(ns) > 1:
        print("\nРАЗБРОС ПО СИДАМ (std F1)")
        print("-" * 78)
        print(base.pivot_table(index="method", columns="graph", values="f1",
                               aggfunc="std")
                  .reindex(index=order, columns=cols).round(3).to_string())

    print("\nЦЕНА СЛЕПОГО ВЫБОРА: oracle F1 минус выбранное")
    print("-" * 78)
    loss = base.assign(loss=base.oracle_f1 - base.f1)
    print(loss.pivot_table(index="method", columns="graph", values="loss")
              .reindex(index=order, columns=cols).round(3).to_string())

    free = [m for m in order if METHODS[m][1] != "given"]
    if free:
        print("\nНАЙДЕННОЕ ЧИСЛО РЁБЕР против истинного")
        print("-" * 78)
        t = base[base.method.isin(free)].pivot_table(
            index="method", columns="graph", values="n_edges")
        t.loc["ИСТИНА"] = base.groupby("graph")["n_true"].first()
        print(t.reindex(columns=cols).round(0).to_string())

    if df.n.nunique() > 1:
        print("\nЗАВИСИМОСТЬ ОТ ЧИСЛА ОБЪЕКТОВ (F1, среднее по графам и сидам)")
        print("-" * 78)
        print(df.pivot_table(index="method", columns="n", values="f1")
                .reindex(order).round(3).to_string())

    auto = base[base.method == "partial_free"]
    if len(auto) and "score" in auto:
        print("\nЧТО ВЫБРАЛ partial_free")
        print("-" * 78)
        pick = auto.groupby("graph").apply(
            lambda g: (g.score == "partial_aracne").mean(),
            include_groups=False)
        tri = base.groupby("graph")["tri_true"].first()
        for g in cols:
            print(f"  {g:12s} треугольников {int(tri[g]):4d}   "
                  f"ARACNE выбран в {pick.get(g, 0):.0%} прогонов")

    print("\n`corr_static` и `partial_static` знают истинное число рёбер;")
    print("`partial_free` не знает плотность и подбирает её по устойчивости.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="один граф, один сид")
    ap.add_argument("--summary", action="store_true", help="только свести")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--graphs", default=",".join(GRAPHS))
    ap.add_argument("--n-grid", default="500,1000,2000,6000",
                    help="через запятую: сколько объектов в выборке")
    ap.add_argument("--side", type=int, default=12)
    a = ap.parse_args()

    out = results_dir(load_config()) / "structure.csv"
    if a.summary:
        return summarize(out)
    if a.quick:
        graphs, seeds, ns, side = ("grid",), (0,), (3000,), 10
    else:
        graphs = tuple(a.graphs.split(","))
        seeds = tuple(range(a.seeds))
        ns = tuple(int(v) for v in a.n_grid.split(","))
        side = a.side
    rc = run(graphs, seeds, ns, side, out)
    return rc or summarize(out)


if __name__ == "__main__":
    sys.exit(main())
