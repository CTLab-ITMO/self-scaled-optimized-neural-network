import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


def _seed_from_str(base_seed: int, s: str) -> int:
    msg = f"{base_seed}|{s}".encode("utf-8")
    digest = hashlib.blake2b(msg, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


class HashedRandomProjection2D:
    """
    Фиксированная 2D-проекция пространства параметров.
    Не зависит от общего числа параметров и их порядка — только от имени параметра.
    Работает при добавлении/удалении параметров (SSONN growth).
    """
    def __init__(self, base_seed: int = 0, normalize: bool = True, cache: bool = True):
        self.base_seed = int(base_seed)
        self.normalize = normalize
        self.cache = cache
        self._cache: Dict[Tuple[str, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def project(self, model: torch.nn.Module) -> Tuple[float, float]:
        x = 0.0
        y = 0.0
        total = 0

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            w = p.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            n = w.numel()
            if n == 0:
                continue

            # Ключ кэша зависит от имени и размера (shape может меняться)
            key = (name, n, self.base_seed)
            if self.cache and key in self._cache:
                rx, ry = self._cache[key]
            else:
                # Два независимых “случайных” направления по имени параметра
                sx = _seed_from_str(self.base_seed, f"{name}|x|{n}")
                sy = _seed_from_str(self.base_seed, f"{name}|y|{n}")

                gx = torch.Generator(device="cpu"); gx.manual_seed(sx)
                gy = torch.Generator(device="cpu"); gy.manual_seed(sy)

                rx = torch.randn(n, generator=gx, dtype=torch.float32)
                ry = torch.randn(n, generator=gy, dtype=torch.float32)

                if self.cache:
                    self._cache[key] = (rx, ry)

            x += torch.dot(w, rx).item()
            y += torch.dot(w, ry).item()
            total += n

        if self.normalize and total > 0:
            scale = (total ** 0.5)
            x /= scale
            y /= scale

        return x, y


@dataclass
class TrajectoryLogger:
    """Логирует 2D-точки траектории + метрики по шагам."""
    projector: HashedRandomProjection2D
    every: int = 1  # логировать каждые N шагов
    points: List[Tuple[float, float]] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    metrics: List[dict] = field(default_factory=list)

    def maybe_log(self, model: torch.nn.Module, step: int, **metrics):
        if step % self.every != 0:
            return
        pt = self.projector.project(model)
        self.points.append(pt)
        self.steps.append(step)
        self.metrics.append(metrics)

    def break_line(self, step: int):
        self.points.append((math.nan, math.nan))
        self.steps.append(step)
        self.metrics.append({"event": "expand"})



def plot_trajectories_grid(
    runs: Dict[str, TrajectoryLogger],
    projectors: List[HashedRandomProjection2D],
    title: str = "Trajectory projections",
    figsize=(10, 10),
    alpha=0.9,
):
    """
    Строит сетку 2x2 (или больше) проекций.
    Чтобы сравнить оптимизаторы: runs = {"adam": loggerA, "sgd": loggerB, ...}
    projectors — список разных проекций (разные base_seed).
    """
    k = len(projectors)
    ncols = 2
    nrows = (k + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.reshape(-1)

    for i, proj in enumerate(projectors):
        ax = axes[i]
        ax.set_title(f"proj seed={proj.base_seed}")
        ax.grid(True, alpha=0.3)

        for run_name, logger in runs.items():
            # ВАЖНО: чтобы переиспользовать points, нужно логировать каждым proj отдельно.
            # Поэтому тут предполагаем, что runs[run_name] уже соответствует этой проекции.
            pts = torch.tensor(logger.points, dtype=torch.float32)
            if pts.numel() == 0:
                continue

            ax.plot(pts[:, 0].numpy(), pts[:, 1].numpy(), label=run_name, alpha=alpha)
            ax.scatter(pts[0, 0].item(), pts[0, 1].item(), s=25, marker="o")   # start
            ax.scatter(pts[-1, 0].item(), pts[-1, 1].item(), s=35, marker="x") # end

        ax.legend()

    # убрать пустые оси
    for j in range(k, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_time_colored(logger: TrajectoryLogger, title="all history", figsize=(5, 5)):
    """График как 'all history' — точки по времени с градиентом."""
    pts = torch.tensor(logger.points, dtype=torch.float32)
    if pts.numel() == 0:
        raise ValueError("No points logged")

    t = torch.linspace(0, 1, pts.shape[0]).numpy()
    fig = plt.figure(figsize=figsize)
    plt.scatter(pts[:, 0].numpy(), pts[:, 1].numpy(), c=t, s=8)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig



def plot_trajectory_direction(
    points,
    title="Trajectory",
    figsize=(6, 6),
    cmap="viridis",
    linewidth=2.0,
    add_arrows=True,
    arrow_every=50,
    arrow_alpha=0.7,
    show_colorbar=False,
):
    """
    points: list[(x,y)] или np.array shape (N,2)
    Поддерживает разрывы через (nan, nan) — линия рвётся.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        raise ValueError("Empty trajectory")

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Маска валидных точек (не NaN)
    valid = ~np.isnan(pts).any(axis=1)

    segments_all = []
    t_all = []

    idx = np.arange(len(pts), dtype=np.float32)

    start = None
    for i, ok in enumerate(valid):
        if ok and start is None:
            start = i
        if (not ok or i == len(pts) - 1) and start is not None:
            end = i if not ok else i + 1
            chunk = pts[start:end]
            if len(chunk) >= 2:
                seg = np.stack([chunk[:-1], chunk[1:]], axis=1)  # (M-1,2,2)
                segments_all.append(seg)
                # цвет сегмента — по "времени" (середина индексов)
                t = (idx[start:end-1] + idx[start+1:end]) * 0.5
                t_all.append(t)
            start = None

    if segments_all:
        segments_all = np.concatenate(segments_all, axis=0)
        t_all = np.concatenate(t_all, axis=0)

        lc = LineCollection(segments_all, cmap=cmap)
        lc.set_array(t_all)
        lc.set_linewidth(linewidth)
        ax.add_collection(lc)

        if show_colorbar:
            fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04, label="time")

    valid_idx = np.where(valid)[0]
    if len(valid_idx) > 0:
        s = valid_idx[0]
        e = valid_idx[-1]
        ax.scatter(pts[s, 0], pts[s, 1], s=60, marker="o", label="start")
        ax.scatter(pts[e, 0], pts[e, 1], s=80, marker="X", label="end")

    if add_arrows and len(valid_idx) > 2:
        vpts = pts[valid]
        for i in range(0, len(vpts) - 1, arrow_every):
            x0, y0 = vpts[i]
            x1, y1 = vpts[i + 1]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=1.2, alpha=arrow_alpha),
            )

    ax.legend(loc="best")
    plt.tight_layout()
    return fig
