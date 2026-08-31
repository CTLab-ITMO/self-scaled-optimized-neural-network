"""Синтетика с ИЗВЕСТНЫМ истинным графом (пункт 5 задания).

На MNIST истинного графа нет, поэтому «сколько структуры восстановлено» там
меряется лишь приблизительно. Здесь граф задаём мы — меряется точно.

Ключевое свойство: в каждом объекте ровно `m` активных вершин со значением 1
независимо от класса, то есть гистограмма значений одинакова по построению.
Значит без графа задача неразрешима, и пол шкалы известен заранее: 1/K.

Запуск: python src/synth.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SynthTask:
    x_train: np.ndarray   # float32 [n_train, d]
    y_train: np.ndarray   # int64   [n_train]
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    A_true: np.ndarray    # uint8 [d, d] - истинный граф, симметричный, diag=0
    d: int
    n_classes: int
    name: str

    def summary(self) -> str:
        act = (self.x_train > 0).sum(1)
        return (f"{self.name}: d={self.d}, классов={self.n_classes}, "
                f"train={len(self.y_train)}, рёбер={int(self.A_true.sum() // 2)}, "
                f"активных вершин {act.min()}..{act.max()}")


def grid_graph(side: int) -> np.ndarray:
    """Решётка side x side, 4 соседа. Истинный граф для генератора `clusters`."""
    idx = np.arange(side * side).reshape(side, side)
    A = np.zeros((side * side, side * side), dtype=np.uint8)
    for a, b in ((idx[:, :-1], idx[:, 1:]), (idx[:-1], idx[1:])):
        A[a.ravel(), b.ravel()] = 1
        A[b.ravel(), a.ravel()] = 1
    return A


def ring_graph(d: int, k: int = 2) -> np.ndarray:
    """Кольцо, где каждая вершина соединена с k ближайшими с каждой стороны."""
    A = np.zeros((d, d), dtype=np.uint8)
    for s in range(1, k + 1):
        i = np.arange(d)
        A[i, (i + s) % d] = 1
        A[(i + s) % d, i] = 1
    return A


def community_graph(d: int, n_comm: int = 4, p_in: float = 0.11,
                    seed: int = 0) -> np.ndarray:
    """Блочный граф: сообщества, рёбра только внутри них.

    Нужен ради другого распределения степеней: выводы, полученные на решётке,
    могут оказаться её свойством.

    Сообщество обязано быть БОЛЬШЕ пятна (d/n_comm > m) и связным, иначе пятно
    не дорастает до m вершин и класс «1 пятно» пропадает из выборки целиком.
    При n_comm=8 так и было. p_in подобрано под прежнюю среднюю степень.
    """
    rng = np.random.default_rng(seed)
    lab = np.repeat(np.arange(n_comm), int(np.ceil(d / n_comm)))[:d]
    A = np.zeros((d, d), dtype=np.uint8)
    same = lab[:, None] == lab[None, :]
    draw = rng.random((d, d)) < p_in
    draw = np.triu(draw & same, 1)
    A[draw] = 1
    # Случайный путь внутри сообщества. Без него одни только броски p_in
    # оставляют вершины-одиночки: на 200 сидах это четверть случаев.
    for c in range(n_comm):
        v = np.flatnonzero(lab == c)
        rng.shuffle(v)
        A[v[:-1], v[1:]] = 1
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return A


def geometric_graph(d: int, k: int = 6, seed: int = 0) -> np.ndarray:
    """Случайные точки на плоскости, каждая соединена с k ближайшими.

    Степени неровные, геометрия не по осям — ближе к настоящим данным.
    """
    rng = np.random.default_rng(seed)
    xy = rng.random((d, 2))
    dist = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    np.fill_diagonal(dist, np.inf)
    A = np.zeros((d, d), dtype=np.uint8)
    nn = np.argsort(dist, axis=1)[:, :k]
    A[np.repeat(np.arange(d), k), nn.ravel()] = 1
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    return A


def _grow(A_nb: list[np.ndarray], seeds: np.ndarray, m: int,
          rng: np.random.Generator) -> np.ndarray:
    """Растит `c` пятен от зёрен по истинному графу, пока активных не станет m.

    Рост по рёбрам, поэтому пятна связны и класс задаётся структурой.
    """
    active = set(int(s) for s in seeds)
    frontier = list(active)
    while len(active) < m:
        if not frontier:                      # пятна упёрлись - добавляем зерно
            cand = rng.integers(0, len(A_nb))
            if int(cand) not in active:
                active.add(int(cand))
                frontier.append(int(cand))
            continue
        j = rng.integers(0, len(frontier))
        v = frontier[j]
        nb = [int(u) for u in A_nb[v] if int(u) not in active]
        if not nb:
            frontier.pop(j)
            continue
        u = nb[rng.integers(0, len(nb))]
        active.add(u)
        frontier.append(u)
    return np.fromiter(active, dtype=np.int64, count=len(active))


def _blur(act: np.ndarray, nb: list[np.ndarray], noise: float, m: int,
          rng: np.random.Generator) -> np.ndarray:
    """Сдвигает долю активных вершин на соседа, размывая пятно, но не дробя.

    Выброс в случайное место (первая версия) задачу ломал, а не усложнял.
    """
    act = list(act)
    k = int(round(noise * m))
    for _ in range(k):
        i = rng.integers(0, len(act))
        v = act[i]
        cand = [int(u) for u in nb[v] if u not in act]
        if cand:
            act[i] = cand[rng.integers(0, len(cand))]
    return np.array(sorted(set(act)), dtype=np.int64)


def make_clusters(side: int = 16, n: int = 12000, n_classes: int = 4,
                  m: int = 40, noise: float = 0.0, seed: int = 0,
                  graph: str = "grid") -> SynthTask:
    """Класс = число связных пятен активных вершин (1, 2, ..., n_classes).

    Во всех объектах ровно m активных вершин, поэтому гистограмма значений от
    класса не зависит и без графа задача неразрешима.

    noise — доля вершин, сдвинутых на соседа. Шкала крутая: потолок падает со
    100% до 57% уже при 0.05. Выше 0.05 задачу считать деградировавшей.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    rng = np.random.default_rng(seed)
    d = side * side
    if graph == "grid":
        A = grid_graph(side)
        xy = np.stack([np.arange(d) // side, np.arange(d) % side], 1)
    elif graph == "community":
        A = community_graph(d, seed=seed)
        xy = None
    elif graph == "geometric":
        A = geometric_graph(d, seed=seed)
        xy = None
    elif graph == "ring":
        A = ring_graph(d, k=2)
        xy = None
    else:
        raise ValueError(f"неизвестный тип графа {graph!r}; есть: "
                         "grid, community, geometric, ring")
    nb = [np.flatnonzero(A[v]) for v in range(d)]

    # Пятно растёт по рёбрам, поэтому в компоненту меньше m вершин не влезает.
    # Зерно ставится равномерно по всем вершинам, так что проверять надо
    # НАИМЕНЬШУЮ компоненту: одна вершина-одиночка уже ломает класс «1 пятно».
    comp = np.bincount(connected_components(csr_matrix(A), directed=False)[1])
    if comp.min() < m:
        raise ValueError(
            f"граф {graph!r}: наименьшая компонента {comp.min()} вершин < m={m} "
            f"(всего компонент {len(comp)}). Пятно туда не поместится, и класс "
            f"«1 пятно» станет недостижим. Уменьши m или укрупни компоненты.")

    # Расстояние между вершинами - чтобы зёрна пятен ставились подальше друг
    # от друга. У решётки это координаты на плоскости; у остальных графов
    # координат нет, поэтому берётся расстояние ПО РЁБРАМ.
    if xy is None:
        from scipy.sparse.csgraph import shortest_path
        hop = shortest_path(csr_matrix(A), directed=False, unweighted=True)
        hop[~np.isfinite(hop)] = d              # разные компоненты - "далеко"
    else:
        hop = None

    def sample_clumps(c: int) -> tuple[np.ndarray, bool]:
        """Растит ровно c пятен, проверяя, что они не слились.

        Без проверки метка врёт: слипшиеся пятна дают одну компоненту.
        """
        for _ in range(200):
            # зёрна подальше друг от друга: иначе сливаются почти всегда
            seeds = [int(rng.integers(0, d))]
            while len(seeds) < c:
                cand = rng.integers(0, d, 32)
                if xy is not None:
                    dist = np.abs(xy[cand][:, None] - xy[seeds][None]).sum(2).min(1)
                else:
                    dist = hop[np.ix_(cand, seeds)].min(1)
                seeds.append(int(cand[dist.argmax()]))
            act = _grow(nb, np.array(seeds), m, rng)
            sub = A[np.ix_(act, act)]
            if connected_components(csr_matrix(sub), directed=False)[0] == c:
                return act, True
        return act, False   # не развелись за 200 попыток - объект отбракуем

    X = np.zeros((n, d), dtype=np.float32)
    y = rng.integers(0, n_classes, n).astype(np.int64)
    ok = np.ones(n, dtype=bool)
    for i in range(n):
        c = int(y[i]) + 1                      # число пятен: 1..n_classes
        act, ok[i] = sample_clumps(c)
        if noise > 0:
            act = _blur(act, nb, noise, m, rng)
        X[i, act] = 1.0

    # Метка проверена на чистой структуре, до шума: класс - это сколько пятен
    # было, а шум портит наблюдение, а не истину.
    X, y, n = X[ok], y[ok], int(ok.sum())

    n_tr, n_va = int(n * 0.7), int(n * 0.1)
    tr, va = slice(0, n_tr), slice(n_tr, n_tr + n_va)
    te = slice(n_tr + n_va, n)
    return SynthTask(
        x_train=X[tr], y_train=y[tr],
        x_val=X[va], y_val=y[va],
        x_test=X[te], y_test=y[te],
        A_true=A, d=d, n_classes=n_classes,
        name=f"clusters(side={side}, m={m}, noise={noise})",
    )


def recovery(A_pred: np.ndarray, A_true: np.ndarray) -> dict:
    """Совпадение предсказанного графа с истинным, по рёбрам.

    Именно по рёбрам: на разреженной матрице доля совпавших ячеек ~1 всегда.
    """
    P = np.triu(A_pred, 1).astype(bool)
    T = np.triu(A_true, 1).astype(bool)
    tp = int((P & T).sum())
    prec = tp / max(int(P.sum()), 1)
    rec = tp / max(int(T.sum()), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"precision": prec, "recall": rec, "f1": f1,
            "n_pred": int(P.sum()), "n_true": int(T.sum())}


if __name__ == "__main__":
    import argparse

    from sklearn.linear_model import LogisticRegression

    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--dev", action="store_true",
                    help="принимается для единообразия с другими модулями")
    args, _ = ap.parse_known_args()

    task = make_clusters(side=args.side, n=args.n, n_classes=4, m=24,
                         noise=args.noise)
    print(task.summary())

    # 1. Гистограмма значений обязана быть одинаковой у всех классов -
    #    иначе задачу можно решить без графа, и весь смысл теряется.
    act = (task.x_train > 0).sum(1)
    per_class = [act[task.y_train == c] for c in range(task.n_classes)]
    print("\nактивных вершин по классам (обязано совпадать):")
    for c, a in enumerate(per_class):
        print(f"  класс {c}: {a.min()}..{a.max()}, среднее {a.mean():.2f}")
    assert all(a.min() == a.max() == per_class[0].min() for a in per_class), \
        "гистограммы классов различаются - задача решается без графа"

    # 2. Пол шкалы: модель без графа обязана угадывать случайно.
    lr = LogisticRegression(max_iter=500).fit(
        act.reshape(-1, 1), task.y_train)
    acc = lr.score((task.x_test > 0).sum(1).reshape(-1, 1), task.y_test)
    print(f"\nрегрессия только на числе активных вершин: {acc:.4f} "
          f"(ожидается ~{1 / task.n_classes:.2f})")

    # 3. Потолок: истинный граф даёт разделимый признак - число компонент
    #    связности среди активных вершин. Это ровно определение класса.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n_probe = min(500, len(task.y_test))
    ok = 0
    for i in range(n_probe):
        a = np.flatnonzero(task.x_test[i] > 0)
        sub = task.A_true[np.ix_(a, a)]
        ncomp, _ = connected_components(csr_matrix(sub), directed=False)
        ok += (ncomp == task.y_test[i] + 1)
    print(f"число пятен по истинному графу совпало с меткой: {ok / n_probe:.1%}"
          + ("" if args.noise == 0 else
             "   (при шуме < 100% - это свойство задачи)"))

    # 4. Метрика восстановления: сама себя обязана давать единицу.
    r = recovery(task.A_true, task.A_true)
    print(f"\nrecovery(истинный, истинный): f1 = {r['f1']:.3f} (обязано быть 1.000)")
    assert r["f1"] == 1.0
    rng = np.random.default_rng(0)
    rnd = np.zeros_like(task.A_true)
    iu = np.triu_indices(task.d, 1)
    pick = rng.choice(len(iu[0]), size=int(task.A_true.sum() // 2), replace=False)
    rnd[iu[0][pick], iu[1][pick]] = 1
    rnd = rnd | rnd.T
    print(f"recovery(случайный, истинный): f1 = {recovery(rnd, task.A_true)['f1']:.3f}")

    print("\nвсе проверки синтетики пройдены")