import argparse
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import load_config
from data import load_data
import graphs as G

LS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40, 50]


def hops(A: np.ndarray, ls: list[int], target: float | None = None,
         lmax: int = 60) -> tuple[dict[int, float], int]:
    """Средняя по вершинам зона охвата: сколько вершин достижимо за <= L рёбер.

    Считается включая саму вершину (self-loop в Â), поэтому L=0 дало бы 1.0.

    ВНИМАНИЕ: аналитическая формула 2L^2+2L+1 для решётки годится только пока
    шар не упёрся в край - на 28x28 это примерно до L = 4..5. При L = 10 она
    даёт 221 против фактических 168, при L = 20 - 841 против 468. Цитировать
    можно только измеренные числа из этой таблицы.
    """
    R = (A > 0).astype(np.float32)
    d = len(A)
    cur = np.eye(d, dtype=np.float32)
    out, lstar = {}, -1
    for L in range(1, lmax + 1):
        cur = ((cur @ R) + cur > 0).astype(np.float32)
        m = float(cur.sum(1).mean())
        if L in ls:
            out[L] = m
        if lstar < 0 and target is not None and m >= target:
            lstar = L
        if lstar > 0 and L >= max(ls):
            break
    return out, lstar


def mixing(A: np.ndarray) -> tuple[float, int, int]:
    """|λ2(Â)| на ГИГАНТСКОЙ КОМПОНЕНТЕ, её размер и число компонент.

    На всём графе λ2 = 1 у любого несвязного кандидата - это детектор
    изолированных вершин, а не скорости перемешивания. Мёртвые пиксели MNIST
    делают несвязными все kNN-графы, поэтому считаем на гиганте.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import csr_matrix
    n, lab = connected_components(csr_matrix(A), directed=False)
    sizes = np.bincount(lab)
    giant = np.flatnonzero(lab == sizes.argmax())
    if len(giant) < 2:                      # empty: все вершины изолированы
        return float("nan"), len(giant), n
    Ag = A[np.ix_(giant, giant)]
    Ah = G.normalize_adj(Ag.astype(np.uint8)).astype(np.float64)
    ev = np.linalg.eigvalsh(Ah)
    return float(max(abs(ev[0]), abs(ev[-2]))), len(giant), n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true",
                    help="d=196 вместо 784 - для быстрой проверки, что код "
                         "работает. Пишет в отдельный файл "
                         "depth_precheck_dev.csv, а не в боевой.")
    args = ap.parse_args()

    cfg = load_config(dev=args.dev)
    data = load_data(cfg, dev=args.dev)
    d = cfg.data.d
    rows = []
    for name in G.available():
        if name == "mlp":
            continue
        g = G.build_graph(name, cfg, data)
        l2, giant, ncomp = mixing(g.A)
        # L* ищется сканированием КАЖДОГО L, а не по списку контрольных точек:
        # иначе он округляется вверх до ближайшей точки списка - у grid4
        # выдавало 35 при фактическом 31.
        h, full = hops(g.A, LS, target=0.9 * giant)
        rows.append(dict(graph=name, group=g.group, n_edges=int(g.A.sum() // 2),
                         lambda2_giant=l2, giant=giant, n_components=ncomp,
                         L_global=full,
                         **{f"hops_L{L}": h[L] for L in LS}))
        print(f"  {name:<18} готов", flush=True)

    df = pd.DataFrame(rows).sort_values("hops_L3")
    os.makedirs("results", exist_ok=True)
    out_name = "results/depth_precheck_dev.csv" if args.dev else "results/depth_precheck.csv"
    df.to_csv(out_name, index=False)

    print("\n" + "=" * 78)
    print("ЗОНА ОХВАТА: сколько из %d вершин видно за L шагов" % d)
    print("=" * 78)
    cols = [1, 2, 3, 5, 8, 12, 20]
    print(f"{'граф':<18}{'|E|':>7}{'λ2':>7}{'L*':>4}", end="")
    for L in cols:
        print(f"{'L=' + str(L):>7}", end="")
    print("\n" + "-" * (36 + 7 * len(cols)))
    for r in df.itertuples():
        Lg = r.L_global if r.L_global > 0 else 99
        print(f"{r.graph:<18}{r.n_edges:>7}{r.lambda2_giant:>7.3f}{Lg:>4}", end="")
        for L in cols:
            print(f"{getattr(r, f'hops_L{L}'):>7.0f}", end="")
        print()
    print("\nL* - глубина, на которой кандидат охватывает 90% своей гигантской компоненты")
    print("     (99 = не достигает и за 60 слоёв)")
    print("λ2  - на гигантской компоненте; ближе к 1 = медленнее перемешивает")
    print(f"\nсохранено: {out_name}")


if __name__ == "__main__":
    main()
