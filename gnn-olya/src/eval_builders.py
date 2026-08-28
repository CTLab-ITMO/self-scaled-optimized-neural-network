"""Оценка способов построения графа БЕЗ обучения сети (пункт 8 задания).

Две метрики, обе не требуют ни одного прогона GNN:

**Восстановление структуры** — насколько построенный граф совпал с истинным.
Меряется только на синтетике (`synth.py`), потому что на MNIST истинного графа
нет: решётка — разумное предположение, но `knn_corr_k8` её обходит по
качеству, значит считать её эталоном нельзя.

**Устойчивость** — насколько граф меняется, если построить его по другой
половине обучающей выборки. Метод, у которого граф разъезжается от смены
выборки, ловит шум, а не структуру. Считается и на синтетике, и на MNIST.

Обе метрики по рёбрам, а не по матрице целиком: матрица разрежена, и доля
совпавших элементов была бы около единицы у любого разреженного графа.

Запуск:
    python src/eval_builders.py                # синтетика: восстановление
    python src/eval_builders.py --mnist --dev  # MNIST: только устойчивость
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config, results_dir  # noqa: E402
import graphs as G  # noqa: E402
from synth import make_clusters, recovery  # noqa: E402

# Кандидаты, которые строятся ТОЛЬКО по данным. Оракулы сюда не входят: они
# знают координаты, и мерить их «восстановление» бессмысленно - они и есть
# ответ. Случайные контроли включены как пол шкалы.
HONEST = ["knn_corr_k4", "knn_corr_k8", "knn_corr_k16", "thr_corr",
          "emb_learned_k4", "emb_learned_k8", "emb_learned_k16",
          "grad_sim_k4", "grad_sim_k8", "grad_sim_k16"]
BASELINE = ["er_knn8", "regular4"]


class _Adapter:
    """Подсовывает билдерам синтетику вместо MNIST.

    Билдеры ожидают объект с полями x_train/y_train и складывают на него кеши
    (`_corr_cache` и подобные). Отдельный класс нужен, чтобы кеши от разных
    выборок не перемешивались: у каждого среза свой адаптер, свой кеш.
    """

    def __init__(self, x, y, pi=None):
        self.x_train = np.ascontiguousarray(x, dtype=np.float32)
        self.y_train = np.ascontiguousarray(y, dtype=np.int64)
        self.pi = np.arange(x.shape[1]) if pi is None else pi


def _cfg_for(d: int, side: int, base=None):
    cfg = load_config() if base is None else base
    cfg = type(cfg)(cfg)                      # мелкая копия, оригинал не трогаем
    cfg["data"] = type(cfg.data)(cfg.data)
    cfg["data"]["d"] = d
    cfg["data"]["side"] = side
    return cfg


def jaccard(A: np.ndarray, B: np.ndarray) -> float:
    """Доля общих рёбер: |A ∩ B| / |A ∪ B|. Единица = графы совпали."""
    iu = np.triu_indices(A.shape[0], 1)
    a, b = A[iu].astype(bool), B[iu].astype(bool)
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def stability(name: str, x, y, cfg, n_splits: int = 2, seed: int = 0) -> dict:
    """Строит граф по непересекающимся половинам выборки и сравнивает.

    Если метод ловит настоящую структуру, обе половины дадут почти один и тот
    же граф. Если ловит шум - разъедется. Половины именно непересекающиеся:
    на пересекающихся сходство завышено просто общими объектами.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    parts = np.array_split(idx, n_splits)
    built = []
    for p in parts:
        built.append(G.build_graph(name, cfg, _Adapter(x[p], y[p])).A)
    sims = [jaccard(built[i], built[j])
            for i in range(len(built)) for j in range(i + 1, len(built))]

    # ЛОВУШКА, которую метрика обязана ловить сама. Граф, вообще не глядящий
    # на данные (случайный от фиксированного сида, полный, решётка), при смене
    # половины выборки не меняется НИКАК - и получает устойчивость 1.000.
    # То есть метрика в наивном виде награждает за игнорирование данных.
    # Признак: графы по разным половинам совпали побитово.
    # Порог, а не точное равенство: `er_knn8` согласован по числу рёбер с
    # data-driven кандидатом, поэтому от смены половины меняется на пару рёбер
    # и побитово не совпадает - хотя структура у него всё равно случайная.
    data_free = (all(np.array_equal(built[0], b) for b in built[1:])
                 or float(np.mean(sims)) > 0.99)
    return {"stability": float(np.mean(sims)),
            "data_free": bool(data_free),
            "n_edges": int(built[0].sum() // 2)}


def run_synth(side: int, n: int, n_classes: int, m: int, seed: int) -> pd.DataFrame:
    task = make_clusters(side=side, n=n, n_classes=n_classes, m=m, seed=seed)
    print(task.summary(), flush=True)
    cfg = _cfg_for(task.d, side)
    data = _Adapter(task.x_train, task.y_train)

    rows = []
    for name in HONEST + BASELINE:
        t = time.time()
        try:
            A = G.build_graph(name, cfg, data).A
        except Exception as exc:  # noqa: BLE001
            print(f"  {name}: не построился - {exc}", flush=True)
            continue
        r = recovery(A, task.A_true)
        st = stability(name, task.x_train, task.y_train, cfg, seed=seed)
        rows.append(dict(graph=name, **r,
                         stability=np.nan if st["data_free"] else st["stability"],
                         data_free=st["data_free"],
                         build_s=round(time.time() - t, 1)))
        mark = "не смотрит на данные" if st["data_free"] else \
            f"устойчивость={st['stability']:.3f}"
        print(f"  {name:<18} f1={r['f1']:.3f}  {mark}", flush=True)
    return pd.DataFrame(rows)


def run_mnist(dev: bool) -> pd.DataFrame:
    from data import load_data
    cfg = load_config(dev=dev)
    data = load_data(cfg)
    rows = []
    for name in HONEST + BASELINE:
        t = time.time()
        st = stability(name, data.x_train, data.y_train, cfg)
        rows.append(dict(graph=name,
                         stability=np.nan if st["data_free"] else st["stability"],
                         data_free=st["data_free"], n_edges=st["n_edges"],
                         build_s=round(time.time() - t, 1)))
        mark = "не смотрит на данные" if st["data_free"] else \
            f"устойчивость={st['stability']:.3f}"
        print(f"  {name:<18} {mark}  рёбер={st['n_edges']}", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist", action="store_true",
                    help="Считать на MNIST (только устойчивость: истинного "
                         "графа там нет).")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--side", type=int, default=12)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--classes", type=int, default=4)
    ap.add_argument("--m", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mnist:
        print(f"MNIST{' (dev)' if args.dev else ''}: устойчивость графов\n")
        df = run_mnist(args.dev).sort_values("stability", ascending=False)
        out = results_dir(load_config()) / "builders_mnist.csv"
    else:
        print("СИНТЕТИКА: восстановление истинного графа и устойчивость\n")
        df = run_synth(args.side, args.n, args.classes, args.m,
                       args.seed).sort_values("f1", ascending=False)
        out = results_dir(load_config()) / "builders_synth.csv"

    df.to_csv(out, index=False)
    print("\n" + "=" * 72)
    print(df.to_string(index=False))
    print("=" * 72)
    print(f"\nсохранено: {out}")
    if "f1" in df:
        print("\nf1 - совпадение с истинным графом по рёбрам (1.0 = точное)")
    print("устойчивость - совпадение графов, построенных по разным половинам "
          "выборки")
    print("пусто в этой колонке - граф почти не зависит от данных, для него")
    print("      метрика бессмысленна: он тривиально устойчив, и это ничего")
    print("      не значит\n")
    print("ГЛАВНОЕ: устойчивость сама по себе НЕ признак качества. Граф, вообще")
    print("не смотрящий на данные, устойчив идеально. Читать её можно только")
    print("вместе с восстановлением: высокая устойчивость при низком f1 значит,")
    print("что метод стабильно находит не то.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
