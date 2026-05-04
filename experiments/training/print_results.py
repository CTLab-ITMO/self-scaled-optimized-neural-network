#!/usr/bin/env python3
"""Pretty-print results from a (PC-)DARTS + SSONN-tail (cosine) run.

Usage:
    python print_results.py                                       # latest darts_tail_cosine_*.pt in ./runs
    python print_results.py --pcdarts                             # latest pcdarts_tail_cosine_*.pt
    python print_results.py runs/darts_tail_cosine_<stamp>.pt     # explicit path
    python print_results.py --runs-dir /elsewhere/runs --pcdarts
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

INITIAL_TAIL = 576 * 10 + 10


# Lenient unpickler — synthesises stub classes for anything pickle can't import,
# so the print script never needs the model class definitions on sys.path.
class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)

    def __repr__(self):
        return f"<_Stub {self.__class__.__name__}>"


class _LenientUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError, ModuleNotFoundError, KeyError):
            return type(name, (_Stub,), {})


class _LenientPickleModule:
    Unpickler = _LenientUnpickler
    UnpicklingError = pickle.UnpicklingError


def _try_setup_backbone_sys_path(ckpt_path: Path) -> str | None:
    """Add the appropriate backbone directory to sys.path so the Genotype
    namedtuple unpickles into its real class."""
    name = ckpt_path.name
    if "pcdarts_tail" in name:
        rel = "../../../../PC-DARTS-master"
    elif "darts_tail" in name:
        rel = "../../../../darts-master/cnn"
    else:
        return None
    candidate = (ckpt_path.parent / rel).resolve()
    if candidate.is_dir() and (candidate / "genotypes.py").exists():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        return str(candidate)
    return None


def section(title: str, char: str = "=", width: int = 78) -> None:
    bar = char * width
    print(f"\n{bar}\n  {title}\n{bar}")


def subsection(title: str, char: str = "-", width: int = 78) -> None:
    print(f"\n  {title}\n  {char * (width - 2)}")


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_pct(x: float | None, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def fmt_seconds(s: float | None) -> str:
    if s is None:
        return "n/a"
    h, rem = divmod(s, 3600)
    m, _ = divmod(rem, 60)
    if h >= 1:
        return f"{s:.0f}s  ({int(h)}h{int(m):02d}m,  {h + m / 60:.2f}h)"
    return f"{s:.0f}s  ({m:.1f}m)"


def maybe(d: dict, key: str, default: Any = None) -> Any:
    return d.get(key, default) if isinstance(d, dict) else default


def print_header(path: Path, ckpt: dict) -> None:
    section("RUN")
    print(f"  file:         {path}")
    print(f"  size:         {path.stat().st_size / 1e6:.2f} MB")
    print(f"  timestamp:    {maybe(ckpt, 'timestamp', '?')}")
    print(f"  seed:         {maybe(ckpt, 'seed', '?')}")
    print(f"  num_epochs:   {maybe(ckpt, 'num_epochs', '?')}")
    print(f"  train time:   {fmt_seconds(maybe(ckpt, 'train_seconds'))}")
    g = ckpt.get("genotype")
    if g is not None:
        gname = type(g).__name__
        normal = getattr(g, "normal", None)
        reduce = getattr(g, "reduce", None)
        print(f"  genotype:     {gname}")
        if normal:
            print(f"    normal:     {normal}")
        if reduce:
            print(f"    reduce:     {reduce}")


def print_env(env: dict | None) -> None:
    if not env:
        return
    section("ENVIRONMENT")
    for k in ("python", "torch", "cuda", "cudnn", "numpy", "device_name", "platform"):
        if k in env:
            print(f"  {k:<13} {env[k]}")


def print_final_metrics(ckpt: dict) -> None:
    section("FINAL METRICS")
    test_acc = ckpt.get("test_accuracy")
    print(f"  test  top-1:  {fmt_pct(test_acc)}    error:  {fmt_pct(None if test_acc is None else 1 - test_acc)}")
    print(f"  test  top-5:  {fmt_pct(ckpt.get('test_top5'))}")
    test_loss = ckpt.get("test_loss")
    print(f"  test  loss:   {test_loss:.4f}" if test_loss is not None else "  test  loss:   n/a")

    val_acc = ckpt.get("val_accuracy")
    val_top5 = ckpt.get("val_top5")
    val_loss = ckpt.get("val_loss")
    if val_acc is not None and (val_acc != test_acc or val_loss != test_loss):
        print()
        print(f"  val   top-1:  {fmt_pct(val_acc)}")
        print(f"  val   top-5:  {fmt_pct(val_top5)}")
        print(f"  val   loss:   {val_loss:.4f}" if val_loss is not None else "  val   loss:   n/a")
    elif val_acc is not None:
        print(f"  (val == test in this regime; same numbers)")


def print_params(p: dict | None) -> None:
    if not p:
        return
    section("PARAMETERS")
    tail_total = p.get("tail_total", 0)
    tail_added = tail_total - INITIAL_TAIL
    total = p.get("total", 0)
    backbone = p.get("backbone", 0)

    print(f"  total:           {fmt_int(total):>14}")
    print(f"  trainable:       {fmt_int(p.get('trainable', 0)):>14}")
    print(f"  backbone:        {fmt_int(backbone):>14}    ({backbone / total * 100:.2f}% of total)" if total else "")
    print(f"  tail_initial:    {fmt_int(INITIAL_TAIL):>14}    Linear(576, 10)")
    print(f"  tail_total:      {fmt_int(tail_total):>14}    ({tail_total / total * 100:.2f}% of total)" if total else "")
    print(f"  tail_added:      {fmt_int(tail_added):>14}    (= tail_total - tail_initial)")
    print(f"  tail_non_zero:   {fmt_int(p.get('tail_non_zero', 0)):>14}")
    print(f"  tail_sparsity:   {fmt_pct(p.get('tail_sparsity'), 3):>14}")
    print(f"  expansions:      {p.get('expansions', 0):>14}")


def print_arch(ckpt: dict) -> None:
    section("GROWN TAIL ARCHITECTURE")
    main = ckpt.get("fc0_main_shape")
    layout = ckpt.get("fc0_sub_layout", []) or []
    print(f"  main projection (out, in_grown):  {main}")
    if not layout:
        print(f"  sub_linears: none (no expansions performed)")
        return
    print(f"  sub_linears  ({len(layout)}):")
    print(f"    {'idx':>4}  {'in_dim':>8}  {'out_dim':>8}  {'params':>14}")
    cum = 0
    for i, (in_dim, out_dim) in enumerate(layout):
        params = in_dim * out_dim + out_dim
        cum += params
        print(f"    {i:>4}  {in_dim:>8}  {out_dim:>8}  {fmt_int(params):>14}")
    print(f"    {'sum':>4}  {'':>8}  {'':>8}  {fmt_int(cum):>14}")


def print_per_class(ckpt: dict) -> None:
    pc = ckpt.get("test_per_class")
    if not pc:
        return
    section("PER-CLASS TEST METRICS")
    print(f"  {'class':<14} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}")
    print(f"  {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for i in range(10):
        cn = CIFAR10_CLASSES[i]
        d = pc.get(str(i)) or pc.get(cn) or {}
        if not d:
            continue
        print(
            f"  {cn:<14} "
            f"{d.get('precision', 0):>10.4f} "
            f"{d.get('recall', 0):>10.4f} "
            f"{d.get('f1-score', 0):>10.4f} "
            f"{int(d.get('support', 0)):>10,}"
        )
    print(f"  {'-' * 14} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")
    for label_in, label_out in (("accuracy", "accuracy"), ("macro avg", "macro avg"), ("weighted avg", "weighted avg")):
        d = pc.get(label_in)
        if not d:
            continue
        if isinstance(d, float):
            print(f"  {label_out:<14} {'':>10} {'':>10} {d:>10.4f} {'':>10}")
        else:
            print(
                f"  {label_out:<14} "
                f"{d.get('precision', 0):>10.4f} "
                f"{d.get('recall', 0):>10.4f} "
                f"{d.get('f1-score', 0):>10.4f} "
                f"{int(d.get('support', 0)):>10,}"
            )


def print_confusion(ckpt: dict) -> None:
    cm = ckpt.get("test_confusion")
    if not cm:
        return
    section("CONFUSION MATRIX  (rows = true, cols = predicted)")
    cm_arr = np.asarray(cm, dtype=int)
    n = cm_arr.shape[0]
    cell = 6
    header_label_w = 14
    print(" " * (header_label_w + 2) + "".join(f"{CIFAR10_CLASSES[c][:cell-1]:>{cell}}" for c in range(n)))
    print(" " * (header_label_w + 2) + "-" * (cell * n))
    for r in range(n):
        row = cm_arr[r]
        cells = "".join(f"{v:>{cell}}" for v in row)
        print(f"  {CIFAR10_CLASSES[r]:<{header_label_w}}{cells}    sum={row.sum():>5,}")
    diag = cm_arr.trace()
    total = cm_arr.sum()
    if total:
        print(f"\n  diagonal / total: {diag:,} / {total:,} = {diag / total * 100:.2f}%")


def print_hyperparams(hp: dict | None) -> None:
    if not hp:
        return
    section("HYPERPARAMETERS")
    keys = (
        "num_epochs", "batch_size", "learning_rate", "weight_decay",
        "t_max", "task_type",
        "edge_score_aggregation", "edge_importance_metric",
        "expansion_thresholds", "max_new_edges_per_expansion",
        "min_epochs_between_expansions", "plateau_threshold", "plateau_window_size",
        "diff_threshold", "prune_after_epochs", "pruning_thresholds",
        "start_fully_connected",
    )
    for k in keys:
        if k not in hp:
            continue
        v = hp[k]
        s = repr(v) if not isinstance(v, (int, float, str, bool, dict, list, tuple, type(None))) else v
        print(f"  {k:<32} {s}")


def print_snapshots(ckpt: dict, runs_dir: Path) -> None:
    snaps = ckpt.get("expansion_snapshots") or []
    if not snaps:
        return
    section("PRE-EXPANSION SNAPSHOTS")
    for i, name in enumerate(snaps):
        p = runs_dir / name
        if not p.exists():
            print(f"  [{i}] {name}  (file missing)")
            continue
        try:
            s = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [{i}] {name}  (load failed: {e})")
            continue
        subsection(f"[{i}] {name}    ({p.stat().st_size / 1e6:.1f} MB)")
        print(f"    expansion_index:  {s.get('expansion_index', '?')}")
        print(f"    epoch_estimate:   {s.get('epoch_estimate', '?')}")
        val_acc = s.get("val_accuracy")
        val_loss = s.get("val_loss")
        print(f"    val_acc:          {fmt_pct(val_acc)}")
        print(f"    val_loss:         {val_loss:.4f}" if val_loss is not None else "    val_loss:         n/a")
        print(f"    fc0_main_shape:   {s.get('fc0_main_shape')}")
        print(f"    fc0_sub_layout:   {s.get('fc0_sub_layout')}")
        em = s.get("edge_metric") or {}
        for ln, ed in em.items():
            print(f"    edge_metric[{ln!r}]:")
            print(f"      shape:           {ed.get('shape')}")
            for stat in ("max", "mean", "std", "min"):
                if stat in ed:
                    print(f"      {stat:<14}   {ed[stat]:.6e}")
            print(f"      threshold:       {s.get('thresholds', {}).get(ln)}")
            print(f"      above_threshold: {ed.get('above_threshold')}")
            vals = ed.get("values_normalized")
            if vals is not None:
                vals = np.asarray(vals)
                print(f"      values quantiles "
                      f"(min, 25%, 50%, 75%, max): "
                      f"{vals.min():.4e}  {np.percentile(vals, 25):.4e}  "
                      f"{np.percentile(vals, 50):.4e}  {np.percentile(vals, 75):.4e}  "
                      f"{vals.max():.4e}")


def print_external_artifacts(ckpt: dict, runs_dir: Path) -> None:
    items = []
    for k in ("test_logits_npy", "val_logits_npy"):
        v = ckpt.get(k)
        if v:
            items.append((k, v))
    snaps = ckpt.get("expansion_snapshots") or []
    if items or snaps:
        section("EXTERNAL ARTIFACTS")
        for k, v in items:
            p = runs_dir / v
            extra = f"  ({p.stat().st_size / 1e6:.2f} MB)" if p.exists() else "  (missing)"
            print(f"  {k:<22} {v}{extra}")
        for s in snaps:
            p = runs_dir / s
            extra = f"  ({p.stat().st_size / 1e6:.2f} MB)" if p.exists() else "  (missing)"
            print(f"  pre_expand_snapshot    {s}{extra}")


def find_latest(runs_dir: Path, prefix: str) -> Path:
    candidates = [
        p for p in runs_dir.glob(f"{prefix}_*.pt")
        if "pre_expand" not in p.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No '{prefix}_*.pt' (excluding pre_expand) found in {runs_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "checkpoint", nargs="?", default=None,
        help="Path to a *_tail_cosine_<stamp>.pt file. If omitted, finds the latest in --runs-dir.",
    )
    parser.add_argument(
        "--pcdarts", action="store_true",
        help="When auto-finding, look for pcdarts_tail_cosine_* instead of darts_tail_cosine_*.",
    )
    parser.add_argument(
        "--runs-dir", default="runs", type=Path,
        help="Directory containing the checkpoints (default: ./runs)",
    )
    args = parser.parse_args()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        prefix = "pcdarts_tail_cosine" if args.pcdarts else "darts_tail_cosine"
        ckpt_path = find_latest(args.runs_dir, prefix)

    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    backbone_added = _try_setup_backbone_sys_path(ckpt_path)
    print(f"\nLoading {ckpt_path} ...")
    if backbone_added:
        print(f"  added to sys.path:  {backbone_added}")
    else:
        print(f"  (no backbone dir found alongside; Genotype will appear as a stub)")

    ckpt = torch.load(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_LenientPickleModule,
    )
    runs_dir = ckpt_path.parent

    print_header(ckpt_path, ckpt)
    print_env(ckpt.get("env"))
    print_final_metrics(ckpt)
    print_params(ckpt.get("params"))
    print_arch(ckpt)
    print_per_class(ckpt)
    print_confusion(ckpt)
    print_hyperparams(ckpt.get("hyperparams"))
    print_snapshots(ckpt, runs_dir)
    print_external_artifacts(ckpt, runs_dir)
    print()


if __name__ == "__main__":
    main()
