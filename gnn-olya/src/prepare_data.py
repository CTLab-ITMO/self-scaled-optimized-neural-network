# скачивает MNIST и сохраняет его в удобном для проекта формате
# обычно запускается один раз перед экспериментами

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

from config import artifacts_dir, load_config

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}

EXPECTED = {
    "x_train": (60000, 784),
    "y_train": (60000,),
    "x_test": (10000, 784),
    "y_test": (10000,),
}


def _download(fname: str, dst: Path) -> Path:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  {fname}: уже скачан")
        return dst
    last_err: Exception | None = None
    for mirror in MIRRORS:
        url = mirror + fname
        try:
            print(f"  {fname}: качаю с {mirror}")
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            dst.write_bytes(data)
            print(f"  {fname}: {len(data) / 1e6:.1f} MB OK")
            return dst
        except Exception as exc:  # noqa: BLE001 - хотим попробовать все зеркала
            print(f"  {fname}: зеркало не ответило ({exc})")
            last_err = exc
    raise RuntimeError(f"не удалось скачать {fname}") from last_err


def _read_idx(path: Path) -> np.ndarray:
    """Разбор IDX-формата. torchvision не нужен - это 15 строк."""
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        n_dims = magic & 0xFF
        dtype_code = (magic >> 8) & 0xFF
        if dtype_code != 0x08:
            raise ValueError(f"{path.name}: ожидался uint8, код {dtype_code:#x}")
        shape = struct.unpack(f">{n_dims}I", f.read(4 * n_dims))
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size != int(np.prod(shape)):
        raise ValueError(f"{path.name}: размер не сошёлся {arr.size} vs {shape}")
    return arr.reshape(shape)


def main() -> None:
    cfg = load_config()
    art = artifacts_dir(cfg)
    raw_dir = art / "mnist_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = art / "mnist.npz"

    if out.exists():
        print(f"{out} уже есть, перезаписываю")

    arrays: dict[str, np.ndarray] = {}
    for key, fname in FILES.items():
        path = _download(fname, raw_dir / fname)
        arr = _read_idx(path)
        if arr.ndim == 3:  # изображения: [n, 28, 28] -> [n, 784]
            arr = arr.reshape(arr.shape[0], -1)
        arrays[key] = np.ascontiguousarray(arr)

    for key, shape in EXPECTED.items():
        got = arrays[key].shape
        if got != shape:
            raise ValueError(f"{key}: ожидалась форма {shape}, получена {got}")

    for key in ("y_train", "y_test"):
        labels = arrays[key]
        if labels.min() != 0 or labels.max() != 9:
            raise ValueError(f"{key}: метки вне диапазона 0..9")
        counts = np.bincount(labels, minlength=10)
        print(f"  {key}: распределение классов {counts.tolist()}")

    for key in ("x_train", "x_test"):
        x = arrays[key]
        print(f"  {key}: min={x.min()} max={x.max()} mean={x.mean():.2f}")

    np.savez_compressed(out, **arrays)
    print(f"\nсохранено: {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print("этот файл нужно положить на флешку вместе с кодом")


if __name__ == "__main__":
    main()
