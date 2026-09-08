# загружает MNIST и подготавливает данные для модели
# делит данные на train/validation/test и переставляет признаки как нужно эксперименту

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import Config, artifacts_dir, load_config


@dataclass
class Data:
    x_train: np.ndarray  # float32 [n_train, d], значения 0..1, УЖЕ переставлены
    y_train: np.ndarray  # int64   [n_train]
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    d: int
    side: int
    pi: np.ndarray  # int64 [d], перестановка признаков

    def sizes(self) -> str:
        return (
            f"train {self.x_train.shape}  val {self.x_val.shape}  "
            f"test {self.x_test.shape}  d={self.d} side={self.side}"
        )


def make_permutation(d: int, seed: int) -> np.ndarray:
    """Фиксированная перестановка pi. Одна на все прогоны, кешируется на диск."""
    return np.random.default_rng(seed).permutation(d).astype(np.int64)


def _avg_pool(x: np.ndarray, side: int, pool: int) -> tuple[np.ndarray, int]:
    """Average pooling pool x pool. Нужен только режиму --dev ради скорости."""
    if pool == 1:
        return x, side
    if side % pool != 0:
        raise ValueError(f"side={side} не делится на pool={pool}")
    n = x.shape[0]
    new_side = side // pool
    x = x.reshape(n, new_side, pool, new_side, pool).mean(axis=(2, 4))
    return x.reshape(n, new_side * new_side), new_side


def _stratified_take(y: np.ndarray, pool_idx: np.ndarray, n: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Берёт ровно n индексов из pool_idx, сохраняя баланс классов.

    Метод наибольших остатков - иначе округление по классам не даст ровно n.
    """
    if n > pool_idx.size:
        raise ValueError(f"просят {n} объектов, доступно {pool_idx.size}")
    if n == pool_idx.size:
        return np.sort(pool_idx)

    labels = y[pool_idx]
    classes = np.unique(labels)
    counts = np.array([(labels == c).sum() for c in classes], dtype=np.float64)

    quota = counts / counts.sum() * n
    take = np.floor(quota).astype(np.int64)
    remainder = n - int(take.sum())
    if remainder > 0:
        order = np.argsort(-(quota - take))
        take[order[:remainder]] += 1

    chunks = []
    for c, k in zip(classes, take):
        if k == 0:
            continue
        cls_idx = pool_idx[labels == c]
        chunks.append(rng.permutation(cls_idx)[:k])
    return np.sort(np.concatenate(chunks))


def load_data(cfg: Config | None = None, dev: bool = False) -> Data:
    if cfg is None:
        cfg = load_config(dev=dev)

    npz_path = Path(cfg.root) / cfg.data.npz_path
    if not npz_path.exists():
        raise FileNotFoundError(
            f"нет {npz_path}\nсначала запусти: python src/prepare_data.py"
        )
    raw = np.load(npz_path)

    side_full = int(np.sqrt(raw["x_train"].shape[1]))
    pool = cfg.data.pool

    x_tr_all = raw["x_train"].astype(np.float32) / cfg.data.scale
    x_te_all = raw["x_test"].astype(np.float32) / cfg.data.scale
    y_tr_all = raw["y_train"].astype(np.int64)
    y_te_all = raw["y_test"].astype(np.int64)

    x_tr_all, side = _avg_pool(x_tr_all, side_full, pool)
    x_te_all, _ = _avg_pool(x_te_all, side_full, pool)
    d = side * side
    if d != cfg.data.d:
        raise ValueError(f"d из данных {d} != d из конфига {cfg.data.d}")

    # --- перестановка pi: одна и та же для train/val/test и для всех прогонов
    pi_path = artifacts_dir(cfg) / f"pi_d{d}_seed{cfg.data.permutation_seed}.npy"
    if pi_path.exists():
        pi = np.load(pi_path)
    else:
        pi = make_permutation(d, cfg.data.permutation_seed)
        np.save(pi_path, pi)
    x_tr_all = x_tr_all[:, pi]
    x_te_all = x_te_all[:, pi]

    # --- разбиение: сначала полное train/val, потом (в dev) подвыборка из него,
    #     чтобы dev-выборка была честным подмножеством боевой
    rng = np.random.default_rng(cfg.data.split_seed)
    all_idx = np.arange(y_tr_all.size)

    val_idx = _stratified_take(y_tr_all, all_idx, cfg.data.full_n_val, rng)
    mask = np.ones(y_tr_all.size, dtype=bool)
    mask[val_idx] = False
    train_idx = all_idx[mask]

    if cfg.data.n_val < val_idx.size:
        val_idx = _stratified_take(y_tr_all, val_idx, cfg.data.n_val, rng)
    if cfg.data.n_train < train_idx.size:
        train_idx = _stratified_take(y_tr_all, train_idx, cfg.data.n_train, rng)

    test_idx = np.arange(y_te_all.size)
    if cfg.data.n_test < test_idx.size:
        test_idx = _stratified_take(y_te_all, test_idx, cfg.data.n_test, rng)

    if np.intersect1d(train_idx, val_idx).size:
        raise AssertionError("train и val пересекаются")

    return Data(
        x_train=np.ascontiguousarray(x_tr_all[train_idx]),
        y_train=y_tr_all[train_idx],
        x_val=np.ascontiguousarray(x_tr_all[val_idx]),
        y_val=y_tr_all[val_idx],
        x_test=np.ascontiguousarray(x_te_all[test_idx]),
        y_test=y_te_all[test_idx],
        d=d,
        side=side,
        pi=pi,
    )


if __name__ == "__main__":
    import sys

    dev = "--dev" in sys.argv
    cfg = load_config(dev=dev)
    data = load_data(cfg)

    print(f"режим: {'dev' if dev else 'full'}   hash конфига: {cfg.hash}")
    print(data.sizes())

    # pi - настоящая перестановка?
    assert np.array_equal(np.sort(data.pi), np.arange(data.d)), "pi не перестановка"
    print(f"pi: ок, первые 10 -> {data.pi[:10].tolist()}")

    for name, y in (("train", data.y_train), ("val", data.y_val),
                    ("test", data.y_test)):
        counts = np.bincount(y, minlength=10)
        print(f"{name:5s} классы: {counts.tolist()}  (min/max {counts.min()}/{counts.max()})")

    # Мёртвые пиксели - та самая ловушка из docs/PLAN.md п.3: у них std = 0,
    # корреляция с ними даст NaN, и в kNN-графах они станут изолированными.
    std = data.x_train.std(axis=0)
    dead = int((std == 0).sum())
    near_dead = int((std < 1e-3).sum())
    print(f"мёртвых пикселей (std == 0): {dead} из {data.d}")
    print(f"почти мёртвых (std < 1e-3):  {near_dead} из {data.d}")

    print(f"диапазон значений: {data.x_train.min():.3f} .. {data.x_train.max():.3f}")
