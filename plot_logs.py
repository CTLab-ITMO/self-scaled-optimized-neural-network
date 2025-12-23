#!/usr/bin/env python3
"""
Построение графиков train/val/test loss, accuracy и количества параметров (p ...)
по логам из:
- plain text файла
- .ipynb (извлекает stdout/stderr stream outputs и строит по ним)
 - metrics.csv (который пишет train_sparse_vit.py / train_dense_vit)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


EPOCH_RE = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
PHASE_RE = re.compile(r"---\s*Cur phase:\s*(train|test|val)")
LOSS_RE = re.compile(r"^(train|test|val)\s+loss:\s*([0-9]*\.?[0-9]+)")
ACC_RE = re.compile(r"\bacc:\s*([0-9]*\.?[0-9]+)")
P_RE = re.compile(r"^\s*p\s+(\d+)\s*$")
MODEL_PARAMS_RE = re.compile(r"Model params:\s*(\d+)")


def _read_text_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _extract_lines_from_ipynb(ipynb_path: Path) -> List[str]:
    nb = json.loads(ipynb_path.read_text(encoding="utf-8", errors="ignore"))
    lines: List[str] = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []) or []:
            out_type = out.get("output_type")
            if out_type == "stream":
                # Jupyter stores `text` as list[str] (each usually ends with \n)
                for t in out.get("text", []) or []:
                    lines.extend(str(t).splitlines())
            elif out_type in ("execute_result", "display_data"):
                # Иногда текст лежит в data.text/plain
                data = out.get("data", {}) or {}
                txt = data.get("text/plain")
                if isinstance(txt, list):
                    for t in txt:
                        lines.extend(str(t).splitlines())
                elif isinstance(txt, str):
                    lines.extend(txt.splitlines())
            elif out_type == "error":
                # traceback тоже может быть полезен
                tb = out.get("traceback", []) or []
                for t in tb:
                    lines.extend(str(t).splitlines())
    return lines


@dataclass
class Series:
    epochs: List[int] = field(default_factory=list)
    train_loss: List[Optional[float]] = field(default_factory=list)
    val_loss: List[Optional[float]] = field(default_factory=list)
    test_loss: List[Optional[float]] = field(default_factory=list)
    train_acc: List[Optional[float]] = field(default_factory=list)
    val_acc: List[Optional[float]] = field(default_factory=list)
    test_acc: List[Optional[float]] = field(default_factory=list)
    params: List[Optional[int]] = field(default_factory=list)

    def _ensure_epoch(self, epoch: int) -> int:
        if epoch not in self.epochs:
            self.epochs.append(epoch)
            self.epochs.sort()
        idx = self.epochs.index(epoch)
        self._ensure_len(idx + 1)
        return idx

    def _ensure_len(self, n: int) -> None:
        def pad(lst, fill=None):
            while len(lst) < n:
                lst.append(fill)

        pad(self.train_loss)
        pad(self.val_loss)
        pad(self.test_loss)
        pad(self.train_acc)
        pad(self.val_acc)
        pad(self.test_acc)
        pad(self.params)


def parse_training_logs(lines: Iterable[str]) -> Tuple[Series, str]:
    s = Series()
    cur_epoch: Optional[int] = None
    cur_phase: Optional[str] = None
    total_epochs: Optional[int] = None

    # Подпись (если найдем)
    header_hint: List[str] = []

    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            continue

        m = MODEL_PARAMS_RE.search(line)
        if m:
            header_hint.append(f"Model params: {m.group(1)}")

        m = EPOCH_RE.search(line)
        if m:
            cur_epoch = int(m.group(1))
            total_epochs = int(m.group(2))
            s._ensure_epoch(cur_epoch)
            cur_phase = None
            continue

        m = PHASE_RE.search(line)
        if m:
            cur_phase = m.group(1)
            continue

        m = LOSS_RE.search(line)
        if m and cur_epoch is not None:
            idx = s._ensure_epoch(cur_epoch)
            ph = m.group(1)
            loss = float(m.group(2))
            if ph == "train":
                s.train_loss[idx] = loss
            elif ph == "val":
                s.val_loss[idx] = loss
            else:
                s.test_loss[idx] = loss
            continue

        m = ACC_RE.search(line)
        if m and cur_epoch is not None and cur_phase is not None:
            idx = s._ensure_epoch(cur_epoch)
            acc = float(m.group(1))
            if cur_phase == "train":
                s.train_acc[idx] = acc
            elif cur_phase == "val":
                s.val_acc[idx] = acc
            else:
                s.test_acc[idx] = acc
            continue

        m = P_RE.match(line)
        if m and cur_epoch is not None:
            idx = s._ensure_epoch(cur_epoch)
            s.params[idx] = int(m.group(1))
            continue

    title = " | ".join(dict.fromkeys(header_hint))  # remove duplicates preserving order
    if total_epochs is not None:
        title = (title + " | " if title else "") + f"epochs: {total_epochs}"
    return s, title


def parse_metrics_csv(csv_path: Path) -> Tuple[Series, str]:
    """
    Parse metrics.csv produced by train_sparse_vit.py / train_dense_vit.

    Expected columns:
      epoch, phase in {train,val,test}, loss, acc, params
    """
    s = Series()
    header_hint: List[str] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(float(row.get("epoch", "")))
            except Exception:
                continue

            phase = (row.get("phase") or "").strip().lower()
            if phase not in {"train", "val", "test"}:
                continue

            idx = s._ensure_epoch(epoch)

            def _to_float(v) -> Optional[float]:
                try:
                    if v is None:
                        return None
                    vv = str(v).strip()
                    if vv == "":
                        return None
                    return float(vv)
                except Exception:
                    return None

            loss = _to_float(row.get("loss"))
            acc = _to_float(row.get("acc"))
            params = row.get("params")
            try:
                params_i = int(str(params).strip()) if params is not None and str(params).strip() != "" else None
            except Exception:
                params_i = None

            if loss is not None:
                if phase == "train":
                    s.train_loss[idx] = loss
                elif phase == "val":
                    s.val_loss[idx] = loss
                else:
                    s.test_loss[idx] = loss

            if acc is not None:
                if phase == "train":
                    s.train_acc[idx] = acc
                elif phase == "val":
                    s.val_acc[idx] = acc
                else:
                    s.test_acc[idx] = acc

            # params is same for all phases in our writer; keep latest non-empty
            if params_i is not None:
                s.params[idx] = params_i

    if s.epochs:
        header_hint.append(f"metrics: {csv_path.name}")
    title = " | ".join(header_hint)
    return s, title


def _plot_series(ax, x, y, label: str):
    xs = [xi for xi, yi in zip(x, y) if yi is not None]
    ys = [yi for yi in y if yi is not None]
    if xs:
        ax.plot(xs, ys, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Путь к .log/.txt или .ipynb")
    ap.add_argument("--outdir", default=".", help="Куда сохранить png")
    ap.add_argument("--prefix", default="run", help="Префикс имен файлов")
    ap.add_argument("--dump-log", action="store_true", help="Сохранить извлеченный текстовый лог рядом")
    args = ap.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".csv":
        series, title = parse_metrics_csv(in_path)
    else:
        if in_path.suffix.lower() == ".ipynb":
            lines = _extract_lines_from_ipynb(in_path)
        else:
            lines = _read_text_lines(in_path)

        if args.dump_log:
            dump_path = outdir / f"{args.prefix}_extracted.log"
            dump_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        series, title = parse_training_logs(lines)

    # Lazy import matplotlib (чтобы скрипт мог хотя бы парсить без него)
    import matplotlib.pyplot as plt

    # Loss
    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_series(ax, series.epochs, series.train_loss, "train loss")
    _plot_series(ax, series.epochs, series.val_loss, "val loss")
    _plot_series(ax, series.epochs, series.test_loss, "test loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(("Loss" + (f" ({title})" if title else ""))[:120])
    fig.tight_layout()
    loss_png = outdir / f"{args.prefix}_loss.png"
    fig.savefig(loss_png, dpi=160)
    plt.close(fig)

    # Accuracy
    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_series(ax, series.epochs, series.train_acc, "train acc")
    _plot_series(ax, series.epochs, series.val_acc, "val acc")
    _plot_series(ax, series.epochs, series.test_acc, "test acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(("Accuracy" + (f" ({title})" if title else ""))[:120])
    fig.tight_layout()
    acc_png = outdir / f"{args.prefix}_accuracy.png"
    fig.savefig(acc_png, dpi=160)
    plt.close(fig)

    # Params
    fig, ax = plt.subplots(figsize=(10, 4))
    _plot_series(ax, series.epochs, series.params, "params")
    ax.set_xlabel("epoch")
    ax.set_ylabel("n_params")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(("Parameters" + (f" ({title})" if title else ""))[:120])
    fig.tight_layout()
    p_png = outdir / f"{args.prefix}_params.png"
    fig.savefig(p_png, dpi=160)
    plt.close(fig)

    print("Saved:")
    print(f"  {loss_png}")
    print(f"  {acc_png}")
    print(f"  {p_png}")
    if args.dump_log:
        print(f"  {outdir / f'{args.prefix}_extracted.log'}")


if __name__ == "__main__":
    main()


