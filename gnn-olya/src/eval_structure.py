"""Восстановление истинного графа: сравнение методов без обучения сети.

Отдельно от `refine.py`, потому что сеть здесь не нужна вообще, и прогон
идёт минуты вместо часа. Главный вопрос таблицы:

    обгоняет ли `partial_free`, которому плотность НЕ сообщали,
    методы, которым её подарили.

Колонка «оракул» - подсмотренный в ответ лучший параметр. Нужна ровно для
одного: измерить, сколько стоит слепой выбор. Ни один метод её не использует.

ПРЕДСКАЗАНИЕ, зафиксированное ДО прогона на новых графах: если ARACNE
проваливается из-за треугольников в истинном графе, то на `tree` (их 0) он
обязан выигрывать, на `smallworld` (103) проседать умеренно, на `scalefree`
(121) и `geometric` (673) - сильно. Если порядок не совпадёт, объяснение
через треугольники неверно.

    python src/eval_structure.py                        база: 7 графов, 3 сида
    python src/eval_structure.py --seeds 10             больше сидов
    python src/eval_structure.py --n-grid 500,1000,6000 сколько нужно объектов
    python src/eval_structure.py --noise-grid 0,0.02,0.05  устойчивость к шуму
    python src/eval_structure.py --summary              свести посчитанное
"""

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

# Порядок: сначала без треугольников, потом по их числу вверх. Так провал
# ARACNE читается прямо по строкам таблицы.
GRAPHS = ("tree", "grid", "smallworld", "community", "scalefree",
          "ring", "geometric")

METHODS = {
    "corr_static":    ("corr", "given", "корреляция, плотность задана"),
    "partial_static": ("partial", "given", "частная корр., плотность задана"),
    "aracne_static":  ("partial_aracne", "given", "+ чистка треугольников"),
    "partial_free":   ("auto", "stability", "всё вслепую: оценка, сжатие, плотность"),
    "perm_free":      ("auto", "permutation", "то же, плотность по перестановкам"),
}

KEY = ["graph", "seed", "method", "n", "noise"]


def build(method: str, x: np.ndarray, n_true: int, seed: int):
    score, density, _ = METHODS[method]
    if score == "corr":
        return ST.top_edges(ST.corr_abs(x), n_true), float("nan"), "corr"
    if density == "given":
        f = ST.fit(x, score=score, n_edges=n_true, seed=seed)
    else:
        f = ST.fit(x, score=score, n_edges=None, density=density, seed=seed)
    return f.A, f.shrink, f.score


def oracle_f1(score: str, x: np.ndarray, A_true: np.ndarray, n_true: int) -> float:
    """Лучший достижимый f1 по сетке сжатия, если бы ответ был известен.

    Для `auto` максимум берётся ещё и по видам оценки: иначе оракул считался
    бы для одного варианта, а метод выбирал другой, и «цена слепого выбора»
    выходила отрицательной - метод обгонял бы собственный потолок.
    """
    if score == "corr":
        return recovery(ST.top_edges(ST.corr_abs(x), n_true), A_true)["f1"]
    scores = ("partial", "partial_aracne") if score == "auto" else (score,)
    return max(recovery(ST.top_edges(ST.SCORES[sc](x, sh), n_true),
                        A_true)["f1"] for sc in scores for sh in ST.SHRINK_GRID)


def n_triangles(A: np.ndarray) -> int:
    """Треугольники в графе. Ими объясняется провал ARACNE, поэтому в таблицу."""
    Ai = A.astype(np.int64)
    return int(np.trace(Ai @ Ai @ Ai) // 6)


def run(graphs, seeds, n_grid, noise_grid, side: int, out: Path) -> int:
    import pandas as pd
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(map(tuple, old[KEY].values))
    total = len(graphs) * len(seeds) * len(n_grid) * len(noise_grid) * len(METHODS)
    i = 0
    for g in graphs:
        for s in seeds:
            for n in n_grid:
                for noise in noise_grid:
                    task = None
                    for m in METHODS:
                        i += 1
                        if (g, s, m, n, noise) in done:
                            print(f"[{i}/{total} сделан] {m} / {g} / сид {s} "
                                  f"/ n={n} / шум={noise}", flush=True)
                            continue
                        if task is None:   # генерим один раз на всю пятёрку
                            task = make_clusters(side=side, n=n, n_classes=4,
                                                 m=24, seed=s, graph=g,
                                                 noise=noise)
                        x = task.x_train.astype(np.float64)
                        n_true = int(task.A_true.sum() // 2)
                        t0 = time.time()
                        A, sh, sc = build(m, x, n_true, s)
                        r = recovery(A, task.A_true)
                        row = dict(graph=g, seed=s, method=m, n=n, noise=noise,
                                   n_edges=int(A.sum() // 2), n_true=n_true,
                                   n_train=len(x),
                                   tri_true=n_triangles(task.A_true),
                                   precision=round(r["precision"], 4),
                                   recall=round(r["recall"], 4),
                                   f1=round(r["f1"], 4), shrink=sh, score=sc,
                                   oracle_f1=round(oracle_f1(METHODS[m][0], x,
                                                   task.A_true, n_true), 4),
                                   sec=round(time.time() - t0, 1))
                        print(f"[{i}/{total}] {m:15s} {g:11s} сид {s} n={n:<5d}"
                              f" шум={noise:<5} рёбер {row['n_edges']:4d}/"
                              f"{n_true:<4d} f1={row['f1']:.3f} "
                              f"(оракул {row['oracle_f1']:.3f})  "
                              f"[{row['sec']:.0f} c]", flush=True)
                        df = pd.DataFrame([row])
                        if out.exists():
                            df = pd.concat([pd.read_csv(out), df],
                                           ignore_index=True)
                        out.parent.mkdir(parents=True, exist_ok=True)
                        df.to_csv(out, index=False)
    return 0


def summarize(out: Path) -> int:
    import pandas as pd
    if not out.exists():
        print(f"нет {out} - сначала запусти без --summary")
        return 1
    df = pd.read_csv(out)
    order = [m for m in METHODS if m in set(df.method)]
    base = df[(df.noise == 0) & (df.n == df.n.max())]
    if base.empty:
        base = df

    ns = sorted(df.seed.unique())
    print(f"\nсидов: {len(ns)}   объектов: {sorted(df.n.unique())}   "
          f"шум: {sorted(df.noise.unique())}")

    print("\nВОССТАНОВЛЕНИЕ истинного графа (f1), среднее по сидам")
    print("-" * 78)
    t = base.pivot_table(index="method", columns="graph", values="f1")
    cols = [g for g in GRAPHS if g in t.columns]
    print(t.reindex(index=order, columns=cols).round(3).to_string())
    print("треугольников в истине: " + "  ".join(
        f"{g}={int(base[base.graph == g].tri_true.iloc[0])}" for g in cols))

    if len(ns) > 1:
        print("\nРАЗБРОС ПО СИДАМ (стандартное отклонение f1)")
        print("-" * 78)
        print(base.pivot_table(index="method", columns="graph", values="f1",
                               aggfunc="std")
                  .reindex(index=order, columns=cols).round(3).to_string())

    print("\nЦЕНА СЛЕПОГО ВЫБОРА: оракул минус выбранное")
    print("-" * 78)
    base = base.assign(loss=base.oracle_f1 - base.f1)
    print(base.pivot_table(index="method", columns="graph", values="loss")
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
        print("\nЗАВИСИМОСТЬ ОТ ЧИСЛА ОБЪЕКТОВ (f1, среднее по графам и сидам)")
        print("-" * 78)
        print(df[df.noise == 0].pivot_table(index="method", columns="n",
              values="f1").reindex(order).round(3).to_string())

    if df.noise.nunique() > 1:
        print("\nЗАВИСИМОСТЬ ОТ ШУМА (f1, среднее по графам и сидам)")
        print("-" * 78)
        print(df[df.n == df.n.max()].pivot_table(index="method",
              columns="noise", values="f1").reindex(order).round(3).to_string())

    # Главная проверка компаса: предсказывает ли выбор ARACNE число
    # треугольников. Если объяснение верно, `auto` берёт его только там, где
    # их мало.
    auto = base[base.method == "partial_free"]
    if len(auto) and "score" in auto:
        print("\nЧТО ВЫБРАЛ `auto` (доля прогонов с чисткой треугольников)")
        print("-" * 78)
        pick = auto.groupby("graph").apply(
            lambda g: (g.score == "partial_aracne").mean(), include_groups=False)
        tri = base.groupby("graph")["tri_true"].first()
        for g in cols:
            print(f"  {g:12s} треугольников {int(tri[g]):4d}   "
                  f"взял ARACNE в {pick.get(g, 0):.0%} прогонов")

    print("\nЧитать так: `corr_static`, `partial_static` и `aracne_static` знают")
    print("истинное число рёбер. `partial_free` и `perm_free` - нет, они его")
    print("ищут. Если правило без форы обгоняет правила с форой, значит петля")
    print("по параметрам делает то, чего однократная формула не может.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="один граф, один сид")
    ap.add_argument("--summary", action="store_true", help="только свести")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--graphs", default=",".join(GRAPHS))
    ap.add_argument("--n-grid", default="6000",
                    help="через запятую: сколько объектов в выборке")
    ap.add_argument("--noise-grid", default="0",
                    help="через запятую: доля размытых вершин")
    ap.add_argument("--side", type=int, default=12)
    a = ap.parse_args()

    out = results_dir(load_config()) / "structure.csv"
    if a.summary:
        return summarize(out)
    if a.quick:
        graphs, seeds, ns, noises, side = ("grid",), (0,), (3000,), (0.0,), 10
    else:
        graphs = tuple(a.graphs.split(","))
        seeds = tuple(range(a.seeds))
        ns = tuple(int(v) for v in a.n_grid.split(","))
        noises = tuple(float(v) for v in a.noise_grid.split(","))
        side = a.side
    rc = run(graphs, seeds, ns, noises, side, out)
    return rc or summarize(out)


if __name__ == "__main__":
    sys.exit(main())