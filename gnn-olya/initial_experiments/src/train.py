# обучает одну модель на одном выбранном графе
# после обучения считает качество и сохраняет результат

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from config import Config  # noqa: E402
from data import Data  # noqa: E402
from graphs import META_KEYS, build_graph  # noqa: E402
from metrics import classification_metrics, confusion, mean_pairwise_cosine  # noqa: E402
from model import build_model, n_params, propagation_matrix  # noqa: E402

EVAL_BATCH = 1024
OVERSMOOTH_SAMPLES = 256


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_determinism(seed: int, enabled: bool) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if not enabled:
        return
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] детерминизм недоступен: {exc}")


@torch.no_grad()
def set_tf32(on: bool) -> None:
    """TF32 на матмулах. На Ampere (3060) даёт кратное ускорение больших GEMM.

    Детерминизму не мешает: TF32 урезает мантиссу множителей, накопление
    остаётся в FP32, повторный запуск даёт тот же результат. Но численно это
    ДРУГОЙ режим, поэтому значение пишется в results.csv и входит в ключ
    пропуска - строки FP32 и TF32 нельзя смешивать в одном сравнении.
    Все 110 прогонов первой серии сделаны при tf32 = False.
    """
    torch.backends.cuda.matmul.allow_tf32 = on
    torch.backends.cudnn.allow_tf32 = on


def _evaluate(model, x, y, A_hat, n_classes, prefix) -> tuple[dict, np.ndarray]:
    model.eval()
    preds, loss_sum = [], 0.0
    for i in range(0, x.shape[0], EVAL_BATCH):
        xb, yb = x[i:i + EVAL_BATCH], y[i:i + EVAL_BATCH]
        logits = model(xb, A_hat)
        loss_sum += F.cross_entropy(logits, yb, reduction="sum").item()
        preds.append(logits.argmax(dim=1))
    y_pred = torch.cat(preds).cpu().numpy()
    y_true = y.cpu().numpy()
    out = classification_metrics(y_true, y_pred, n_classes, prefix=prefix)
    out[f"{prefix}loss"] = loss_sum / x.shape[0]
    return out, y_pred


def run_one(graph_name: str, seed: int, cfg: Config, data: Data,
            device: torch.device | None = None,
            verbose: bool = False, on_epoch=None) -> tuple[dict, np.ndarray]:
    """Возвращает (метрики, матрица ошибок на test).

    on_epoch(epoch, train_loss, val_acc) - необязательный колбэк для сбора
    кривой обучения. Нужен калибровке (src/calibrate.py); основной прогон его
    не передаёт, поэтому сигнатура для run_all.py не меняется.
    """
    device = device or get_device()
    is_mlp = graph_name == "mlp"

    # --- граф
    if is_mlp:
        # Те же ключи, что у графов, но пустые: mlp - референс без графа.
        graph_meta = {"group": "R", "honest": True,
                      **{k: float("nan") for k in META_KEYS}}
        A_hat = None
    else:
        graph = build_graph(graph_name, cfg, data)
        graph_meta = {"group": graph.group, "honest": graph.honest, **graph.meta}
        A_hat = torch.from_numpy(
            propagation_matrix(graph.A, cfg.model.kind)).to(device)

    # --- сид: только инициализация весов и порядок батчей
    _set_determinism(seed, cfg.train.deterministic)

    tensors = {}
    for split in ("train", "val", "test"):
        tensors[f"x_{split}"] = torch.from_numpy(
            getattr(data, f"x_{split}")).to(device)
        tensors[f"y_{split}"] = torch.from_numpy(
            getattr(data, f"y_{split}")).to(device)

    model = build_model(graph_name, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)

    n_train = tensors["x_train"].shape[0]
    batch = cfg.train.batch_size
    gen = torch.Generator().manual_seed(seed)

    best_val, best_epoch, best_state = -1.0, -1, None
    t0 = time.time()

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        order = torch.randperm(n_train, generator=gen).to(device)
        epoch_loss = 0.0
        for i in range(0, n_train, batch):
            idx = order[i:i + batch]
            logits = model(tensors["x_train"][idx], A_hat)
            loss = F.cross_entropy(logits, tensors["y_train"][idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * idx.numel()

        val_m, _ = _evaluate(model, tensors["x_val"], tensors["y_val"],
                             A_hat, cfg.model.n_classes, "val_")
        if val_m["val_accuracy"] > best_val:
            best_val = val_m["val_accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if on_epoch is not None:
            on_epoch(epoch, epoch_loss / n_train, val_m["val_accuracy"])
        if verbose:
            print(f"  эпоха {epoch:>3}/{cfg.train.epochs}  "
                  f"train_loss {epoch_loss / n_train:.4f}  "
                  f"val_acc {val_m['val_accuracy']:.4f}"
                  f"{'  <- лучшая' if best_epoch == epoch else ''}",
                  flush=True)  # без flush прогресс не виден при редиректе в файл

    train_time = time.time() - t0

    # Метрики на последней эпохе - чтобы нельзя было сказать, что кандидата
    # недообучили: видно и финальную точку, и лучшую по val.
    last_test, _ = _evaluate(model, tensors["x_test"], tensors["y_test"],
                             A_hat, cfg.model.n_classes, "last_test_")

    model.load_state_dict(best_state)
    test_m, y_pred = _evaluate(model, tensors["x_test"], tensors["y_test"],
                               A_hat, cfg.model.n_classes, "test_")
    val_m, _ = _evaluate(model, tensors["x_val"], tensors["y_val"],
                         A_hat, cfg.model.n_classes, "val_")
    train_m, _ = _evaluate(model, tensors["x_train"], tensors["y_train"],
                           A_hat, cfg.model.n_classes, "train_")

    # Диагностика oversmoothing на фиксированной подвыборке теста
    if is_mlp:
        oversmooth = float("nan")
    else:
        sub = tensors["x_test"][:OVERSMOOTH_SAMPLES]
        emb = model.node_embeddings(sub, A_hat).cpu().numpy()
        oversmooth = mean_pairwise_cosine(emb)

    row = {
        "graph": graph_name,
        "seed": seed,
        "arch": cfg.model.kind,
        "tf32": bool(cfg.train.get("tf32", False)),
        "layers": cfg.model.layers,
        "in_channels": int(getattr(cfg.model, "in_channels", 1) or 1),
        "cfg_hash": cfg.hash,
        "dev": cfg.dev,
        "d": cfg.data.d,
        "n_params": n_params(model),
        "best_epoch": best_epoch,
        "epochs": cfg.train.epochs,
        "train_time_s": round(train_time, 2),
        "device": device.type,
        "node_cosine": oversmooth,
        **{f"graph_{k}": v for k, v in graph_meta.items()},
        **test_m, **val_m, **train_m, **last_test,
    }
    return row, confusion(tensors["y_test"].cpu().numpy(), y_pred,
                          cfg.model.n_classes)


if __name__ == "__main__":
    import argparse

    from config import load_config
    from data import load_data

    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="grid4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--epochs", type=int, default=None,
                    help="ТОЛЬКО для калибровки бюджета обучения. "
                         "В боевых прогонах не использовать: число эпох - "
                         "часть зафиксированного протокола.")
    ap.add_argument("--tf32", action="store_true",
                    help="Включить TF32 на матмулах. На Ampere кратно быстрее, "
                         "детерминизм сохраняется, но численно это другой "
                         "режим: значение пишется в results.csv и входит в "
                         "ключ пропуска. Первая серия (110 прогонов) сделана "
                         "БЕЗ него - не смешивать в одном сравнении.")
    ap.add_argument("--arch", default=None, choices=["gcn"],
                    help="В актуальной серии оставлена только архитектура gcn.")
    ap.add_argument("--in-channels", type=int, default=None, choices=[1],
                    help="В актуальной серии используется один скалярный признак.")
    ap.add_argument("--layers", type=int, default=None,
                    help="Число слоёв L. Ось глубины - предмет исследования "
                         "(см. docs/PREREGISTRATION.md), поэтому L логируется в "
                         "results.csv и входит в ключ пропуска уже посчитанных "
                         "прогонов. Бюджет эпох калибруется на КАЖДОМ L "
                         "отдельно: 150 эпох подобраны при L = 3.")
    ap.add_argument("--n-train", type=int, default=None,
                    help="ТОЛЬКО для калибровки. Позволяет гонять пониженное "
                         "разрешение на полной выборке - тогда число шагов и "
                         "объём данных совпадают с боевым прогоном.")
    args = ap.parse_args()

    cfg = load_config(dev=args.dev)
    cfg.train["tf32"] = bool(args.tf32)
    if args.arch is not None:
        cfg.model["kind"] = args.arch
    if args.in_channels is not None:
        cfg.model["in_channels"] = args.in_channels
    if args.layers is not None:
        cfg.model["layers"] = args.layers
    if args.epochs is not None:
        cfg.train["epochs"] = args.epochs
    if args.n_train is not None:
        cfg.data["n_train"] = args.n_train
    if args.epochs is not None or args.n_train is not None:
        print(f"[!] КАЛИБРОВКА, не боевой прогон: эпох {cfg.train.epochs}, "
              f"train {cfg.data.n_train}\n")
    data = load_data(cfg)
    set_tf32(bool(args.tf32))
    dev = get_device()
    print(f"устройство: {dev}   режим: {'dev' if args.dev else 'full'}   "
          f"d = {cfg.data.d}   hash = {cfg.hash}")
    print(f"кандидат: {args.graph}   сид: {args.seed}   "
          f"арх: {cfg.model.kind}   L = {cfg.model.layers}   "
          f"каналов на входе: {int(getattr(cfg.model, 'in_channels', 1) or 1)}\n")

    row, cm = run_one(args.graph, args.seed, cfg, data, dev, verbose=True)

    print(f"\ntest  accuracy {row['test_accuracy']:.4f}   "
          f"macro-F1 {row['test_macro_f1']:.4f}   loss {row['test_loss']:.4f}")
    print(f"лучшая эпоха {row['best_epoch']}/{row['epochs']}   "
          f"время {row['train_time_s']} с   параметров {row['n_params']}")
    if not np.isnan(row["node_cosine"]):
        print(f"косинус между вершинами {row['node_cosine']:.4f} "
              f"(1.0 = полный коллапс представлений)")
