# строит граф по данным

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

SHRINK_GRID = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85)
DEGREE_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0,
               13.0, 16.0, 20.0, 25.0)



def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Центрирует и нормирует столбцы. Возвращает (z, маска живых признаков)."""
    sd = x.std(0)
    alive = sd > 1e-12
    z = np.zeros_like(x, dtype=np.float64)
    z[:, alive] = (x[:, alive] - x[:, alive].mean(0)) / sd[alive]
    return z, alive


def corr_abs(x: np.ndarray) -> np.ndarray:
    """|корреляция| между признаками. Базовый уровень, он же `corr_static`."""
    d = x.shape[1]
    z, alive = _standardize(x)
    S = np.zeros((d, d))
    if alive.sum() > 1:
        C = np.nan_to_num(np.corrcoef(z[:, alive], rowvar=False), nan=0.0)
        S[np.ix_(alive, alive)] = np.abs(C)
    np.fill_diagonal(S, 0.0)
    return S


def partial_corr(x: np.ndarray, shrink: float = 0.3) -> np.ndarray:
    """|частная корреляция| при условии ВСЕХ остальных признаков.

    Связь через посредника обнуляется - ровно та ошибка, на которой
    сыплется обычная корреляция. Сжатие к единичной матрице обязательно:
    при d, сравнимом с числом объектов, корреляционная матрица почти
    вырождена и обращение без него даёт мусор.
    """
    d = x.shape[1]
    z, alive = _standardize(x)
    S = np.zeros((d, d))
    if alive.sum() < 2:
        return S
    C = np.nan_to_num(np.corrcoef(z[:, alive], rowvar=False), nan=0.0)
    C = (1.0 - shrink) * C + shrink * np.eye(len(C))
    try:
        T = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        T = np.linalg.pinv(C)
    g = np.sqrt(np.clip(np.diag(T), 1e-12, None))
    P = -T / np.outer(g, g)
    np.fill_diagonal(P, 0.0)
    S[np.ix_(alive, alive)] = np.abs(P)
    return S


def aracne(S: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Чистка треугольников: в каждом слабейшее ребро объявляется наведённым.

    Если i-k и j-k сильнее, чем i-j, то связь i-j объяснима через k и
    обнуляется. Стандарт восстановления генных сетей; здесь - второй,
    независимый от обращения матрицы, способ убить связи через посредника.

    Правило предполагает слабую треугольную зависимость и поэтому не должно
    применяться безусловно. В `fit(score="auto")` выбор между обычной частной
    корреляцией и этой чисткой делается по устойчивости на train.
    """
    d = S.shape[0]
    best = np.zeros_like(S)
    # Полный [d, d, d] для d = 784 это 3.8 ГБ, поэтому идём полосами.
    step = max(1, int(4e7 // max(d * d, 1)))
    for a in range(0, d, step):
        b = min(a + step, d)
        best[a:b] = np.minimum(S[a:b, None, :], S[None, :, :]).max(axis=2)
    out = S.copy()
    out[best > S + tol] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


SCORES = {"corr": lambda x, sh: corr_abs(x),
          "partial": partial_corr,
          "partial_aracne": lambda x, sh: aracne(partial_corr(x, sh))}


# ---------------------------------------------------------------------------
#  Граф из оценок
# ---------------------------------------------------------------------------

def top_edges(S: np.ndarray, n_edges: int) -> np.ndarray:
    """Граф из `n_edges` пар с наибольшей оценкой."""
    d = S.shape[0]
    iu = np.triu_indices(d, 1)
    k = int(np.clip(n_edges, 0, len(iu[0])))
    A = np.zeros((d, d), dtype=np.uint8)
    if k:
        top = np.argpartition(-S[iu], k - 1)[:k]
        A[iu[0][top], iu[1][top]] = 1
    return np.maximum(A, A.T)


def n_positive(S: np.ndarray) -> int:
    """Сколько пар вообще получили ненулевую оценку.

    Просить рёбер больше этого числа нельзя: добор идёт парами с оценкой 0,
    то есть по порядку индексов, и он ОДИНАКОВ на любых данных. Устойчивость
    такого графа фиктивно высокая - ровно та же ловушка «награда за
    игнорирование данных», что и у графа, не смотрящего на выборку.
    Всплывает на ARACNE: он обнуляет наведённые связи пачками.
    """
    return int((np.triu(S, 1) > 0).sum())


def jaccard(A: np.ndarray, B: np.ndarray) -> float:
    a = np.triu(A, 1).astype(bool)
    b = np.triu(B, 1).astype(bool)
    return float((a & b).sum() / max((a | b).sum(), 1))


def excess_jaccard(J: float, n_edges: int, d: int) -> float:
    """Жаккар с поправкой на случайное совпадение.

    Без поправки сравнивать устойчивость при РАЗНОМ числе рёбер нельзя:
    два случайных графа на E рёбрах из N пар совпадают на E / (2N - E),
    и при густом графе это само по себе много. Поправка убирает этот фон.
    """
    N = d * (d - 1) // 2
    base = n_edges / max(2 * N - n_edges, 1)
    return float((J - base) / max(1.0 - base, 1e-12))


# ---------------------------------------------------------------------------
#  Селекторы: настройка без знания правильного ответа
# ---------------------------------------------------------------------------

def _halves(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Непересекающиеся половины. На пересекающихся сходство завышено."""
    idx = rng.permutation(n)
    return idx[: n // 2], idx[n // 2:]


def stability(x: np.ndarray, score: str, shrink: float, n_edges: int,
              n_rep: int = 3, seed: int = 0) -> float:
    """Насколько похожи графы, построенные по двум половинам выборки.

    Метод, поймавший настоящую структуру, на обеих половинах даст почти
    один граф. Поймавший шум - разъедется. Правды знать не надо.
    """
    fn = SCORES[score]
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    vals = []
    for _ in range(n_rep):
        a, b = _halves(len(x), rng)
        Sa, Sb = fn(x[a], shrink), fn(x[b], shrink)
        # Потолок нужен и здесь, а не только в select_density: на половине
        # ненулевых оценок бывает меньше, чем на всей выборке, и тогда добор
        # идёт нулями - одинаковыми в обеих половинах, отчего устойчивость
        # фиктивно растёт. Ловушка «награда за игнорирование данных».
        k = min(n_edges, n_positive(Sa), n_positive(Sb))
        if k < 1:
            vals.append(0.0)
            continue
        vals.append(excess_jaccard(jaccard(top_edges(Sa, k),
                                           top_edges(Sb, k)), k, d))
    return float(np.mean(vals))


def select_shrink(x: np.ndarray, n_edges: int, score: str = "partial",
                  grid=SHRINK_GRID, n_rep: int = 3, seed: int = 0):
    """Регуляризация по максимуму устойчивости при фиксированной плотности."""
    vals = [stability(x, score, sh, n_edges, n_rep, seed) for sh in grid]
    return float(grid[int(np.argmax(vals))]), list(map(float, vals))


def select_density(x: np.ndarray, shrink: float, score: str = "partial",
                   grid=DEGREE_GRID, n_rep: int = 3, seed: int = 0,
                   quiet: bool = False):
    """Число рёбер по пику устойчивости с поправкой на случайность.

    Проверено, что пик настоящий, а не край диапазона: устойчивость
    растёт, достигает максимума и снова падает на всех четырёх типах
    графов. Монотонной её сделало бы отсутствие поправки на случайность.
    """
    d = x.shape[1]
    # Потолок - число ненулевых оценок: выше начинается добор нулями, который
    # накручивает устойчивость и уводит выбор в самый густой конец сетки.
    cap = n_positive(SCORES[score](x, shrink))
    cand = sorted({min(cap, max(1, int(round(deg * d / 2)))) for deg in grid})
    vals = [stability(x, score, shrink, E, n_rep, seed) for E in cand]
    # Пик на краю сетки значит, что настоящий максимум за её пределами, и
    # выбранное число рёбер - артефакт диапазона, а не свойство данных.
    k = int(np.argmax(vals))
    if not quiet and len(vals) > 1 and k in (0, len(vals) - 1) and cand[k] < cap:
        print(f"  [!] пик плотности на краю сетки ({cand[k]} рёбер), "
              f"расширь DEGREE_GRID", flush=True)
    return int(cand[int(np.argmax(vals))]), list(map(float, vals)), cand


# ---------------------------------------------------------------------------
#  Сборка: петля по параметрам
# ---------------------------------------------------------------------------

@dataclass
class Fit:
    A: np.ndarray
    shrink: float
    n_edges: int
    score: str
    trace: list = field(default_factory=list)   # (раунд, shrink, E)

    def summary(self) -> str:
        path = " -> ".join(f"(сж {s:.2f}, рёбер {e})" for _, s, e in self.trace)
        return f"{self.score}: {path}"


def select_score(x: np.ndarray, n_edges: int | None = None,
                 scores=("partial", "partial_aracne"), n_rep: int = 3,
                 seed: int = 0) -> str:
    """Выбирает вид оценки по устойчивости её собственного лучшего решения.

    Выбор между двумя вариантами делается тем же train-only критерием
    устойчивости, который используется для остальных ручек.
    """
    best, best_st = scores[0], -np.inf
    for sc in scores:
        f = fit(x, score=sc, n_edges=n_edges, n_rep=n_rep, seed=seed,
                quiet=True)
        st = stability(x, sc, f.shrink, f.n_edges, n_rep=n_rep, seed=seed + 1)
        if st > best_st:
            best, best_st = sc, st
    return best


def fit(x: np.ndarray, score: str = "auto", n_edges: int | None = None,
        rounds: int = 3, n_rep: int = 3, seed: int = 0,
        verbose: bool = False, quiet: bool = False) -> Fit:
    """Полный конвейер: чередуем подбор регуляризации и подбор плотности.

    Именно чередование и делает это итеративным алгоритмом: лучшая
    регуляризация зависит от того, сколько рёбер мы держим, а лучшее число
    рёбер - от регуляризации. Одним махом обе ручки не выставить.

    `n_edges` задают только для честного сравнения с методами, которым
    плотность подарена. Без него алгоритм ищет её сам.
    """
    if score == "auto":
        score = select_score(x, n_edges, n_rep=n_rep, seed=seed)
    d = x.shape[1]
    free = n_edges is None
    E = n_edges if not free else max(1, 2 * d)      # старт: средняя степень 4
    sh = 0.3
    trace = []
    for r in range(rounds):
        sh, _ = select_shrink(x, E, score, n_rep=n_rep, seed=seed)
        if free:
            E, _, _ = select_density(x, sh, score, n_rep=n_rep,
                                     seed=seed, quiet=quiet)
        trace.append((r, sh, E))
        if verbose:
            print(f"  круг {r}: сжатие {sh:.2f}, рёбер {E}", flush=True)
        # Обе ручки перестали двигаться - дальше повторялось бы одно и то же.
        if len(trace) > 1 and trace[-1][1:] == trace[-2][1:]:
            break
    return Fit(A=top_edges(SCORES[score](x, sh), E), shrink=sh, n_edges=E,
               score=score, trace=trace)


# ---------------------------------------------------------------------------
#  Самопроверка
# ---------------------------------------------------------------------------

def _dev() -> int:
    from synth import make_clusters, recovery
    ok = True

    print("=== свойства оценок ===")
    t = make_clusters(side=10, n=1500, n_classes=4, m=12, seed=0, graph="grid")
    x = t.x_train
    for name in SCORES:
        S = SCORES[name](x, 0.3)
        sym = np.allclose(S, S.T)
        dia = not np.any(np.diag(S))
        neg = S.min() >= 0
        print(f"  {name:15s} симметрична {sym}  диагональ нулевая {dia}  "
              f"неотрицательна {neg}")
        ok &= sym and dia and neg

    print("\n=== частная корреляция чинит связи через посредника ===")
    E = int(t.A_true.sum() // 2)
    f_corr = recovery(top_edges(corr_abs(x), E), t.A_true)["f1"]
    f_part = recovery(top_edges(partial_corr(x, 0.15), E), t.A_true)["f1"]
    print(f"  корреляция {f_corr:.3f} -> частная {f_part:.3f}")
    if f_part <= f_corr:
        print("  ПРОВАЛ: частная корреляция не лучше обычной"); ok = False

    print("\n=== селекторы попадают близко к оракулу ===")
    sh, vals = select_shrink(x, E, "partial", seed=0)
    best = max(SHRINK_GRID, key=lambda s:
               recovery(top_edges(partial_corr(x, s), E), t.A_true)["f1"])
    f_sel = recovery(top_edges(partial_corr(x, sh), E), t.A_true)["f1"]
    f_best = recovery(top_edges(partial_corr(x, best), E), t.A_true)["f1"]
    print(f"  сжатие: выбрано {sh} (f1 {f_sel:.3f}), оракул {best} "
          f"(f1 {f_best:.3f}), потеря {f_best - f_sel:.3f}")
    if f_best - f_sel > 0.10:
        print("  ПРОВАЛ: селектор сжатия теряет больше 0.10"); ok = False

    Ehat, dvals, cand = select_density(x, sh, "partial", seed=0)
    print(f"  плотность: выбрано {Ehat} рёбер, истинных {E}")
    interior = 0 < int(np.argmax(dvals)) < len(dvals) - 1
    print(f"  пик внутри диапазона (не на краю): {interior}")
    if not interior:
        print("  ВНИМАНИЕ: пик на границе сетки, расширь DEGREE_GRID")

    print("\n=== полный конвейер вслепую ===")
    got = {}
    for score in ("partial", "partial_aracne"):
        f = fit(x, score=score, seed=0)
        r = recovery(f.A, t.A_true)
        got[score] = r["f1"]
        print(f"  {f.summary()}   f1={r['f1']:.3f}  точн={r['precision']:.3f}  "
              f"полн={r['recall']:.3f}")
        if r["f1"] < 0.3:
            print("  ПРОВАЛ: конвейер вслепую даёт мусор"); ok = False

    # Ветка `auto` - рабочая по умолчанию, и раньше она в самопроверку не
    # входила. Селектор обязан взять вариант не хуже проигравшего: иначе
    # выбор вслепую вредит, а не помогает.
    fa = fit(x, score="auto", seed=0)
    ra = recovery(fa.A, t.A_true)["f1"]
    worse = min(got.values())
    print(f"  auto -> {fa.score}   f1={ra:.3f}   "
          f"(лучший из двух {max(got.values()):.3f}, худший {worse:.3f})")
    if ra < worse - 1e-9:
        print("  ПРОВАЛ: auto выбрал хуже, чем любая из веток"); ok = False
    # Решётка треугольников не содержит, значит чистка обязана выигрывать -
    # это проверка объяснения, а не совпадения чисел.
    if fa.score != "partial_aracne":
        print("  ВНИМАНИЕ: на решётке (0 треугольников) auto не взял чистку")

    print("\nвсе проверки пройдены" if ok else "\nЕСТЬ ПРОВАЛЫ")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="самопроверка")
    a = ap.parse_args()
    sys.exit(_dev() if a.dev else _dev())