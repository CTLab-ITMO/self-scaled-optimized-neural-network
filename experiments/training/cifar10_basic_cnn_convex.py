# CIFAR-10: SSONN tail (cosine-annealing variant) on a basic CNN backbone trained from scratch.
# Run from experiments/training/ of the cosine-annealing SSONN package.

# %%
import copy
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2

SEED = int(os.environ.get("SWEEP_SEED", 0))
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
g = torch.Generator()
g.manual_seed(0)

# --- Distributed setup (DDP via torchrun) ---------------------------------
# Run multi-GPU with: torchrun --nproc_per_node=<N> cifar10_basic_cnn_convex.py
# Without torchrun this falls back to single-process / single-GPU.
import torch.distributed as dist  # noqa: E402

IS_DISTRIBUTED = "RANK" in os.environ and "WORLD_SIZE" in os.environ
if IS_DISTRIBUTED:
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    RANK = int(os.environ["RANK"])
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{LOCAL_RANK}")
else:
    LOCAL_RANK = 0
    WORLD_SIZE = 1
    RANK = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

IS_RANK0 = (RANK == 0)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # speed > bit-exact determinism: let cuDNN pick fastest kernels
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
if IS_RANK0:
    print(f"device: {device}  rank={RANK}/{WORLD_SIZE}  distributed={IS_DISTRIBUTED}")

# AMP config: bfloat16 if supported (no GradScaler needed); else fp16.
USE_AMP = torch.cuda.is_available()
AMP_DTYPE = (
    torch.bfloat16
    if (USE_AMP and torch.cuda.is_bf16_supported())
    else torch.float16
)
print(f"AMP: {USE_AMP}  dtype={AMP_DTYPE}")

# %%
import ssonn.train.train as ssonn_train  # noqa: E402
from ssonn.metrics.edge_finder import EdgeFinder  # noqa: E402, F401
from ssonn.metrics.nonlinearity_metrics import AbsGradientEdgeMetric  # noqa: E402
from ssonn.train.train import (  # noqa: E402
    convert_to_scaled_net,
    eval_one_epoch,
    train_ssonn,
)

# %%
BATCH_SIZE = 256

transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        v2.RandomErasing(p=0.5, scale=(0.1, 0.1), ratio=(1.0, 1.0), value=0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)
transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

full_train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

train_dataset = full_train_dataset
val_dataset = test_dataset  # test set used as val target each epoch

_LOADER_KWARGS = dict(num_workers=8, pin_memory=True, persistent_workers=True)
if IS_DISTRIBUTED:
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True, seed=SEED,
    )
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, **_LOADER_KWARGS)
else:
    train_sampler = None
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g, **_LOADER_KWARGS)
# Val/test/finder run identically on every rank — small datasets, results match.
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, **_LOADER_KWARGS)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, **_LOADER_KWARGS)
finder_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, **_LOADER_KWARGS)


# %%
class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dim only — (B, C, H, W) → permute → LN(C) → permute back.

    Normalizes each (B, H, W) location across C. Per-channel learnable affine.
    Spatial-size-agnostic; no running stats.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        LayerNorm2d(out_channels),
        nn.ReLU(inplace=False),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        LayerNorm2d(out_channels),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2, 2),
    )


class ExpandingHead(nn.Module):
    """Three-block CNN backbone (32→16→8→4) with an SSONN-expandable classifier head."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.backbone = nn.Sequential(
            _conv_block(3, 64),    # 32x32 → 16x16
            _conv_block(64, 128),  # 16x16 →  8x8
            _conv_block(128, 256), #  8x8  →  4x4
            nn.Dropout(p=0.25),
        )
        self.fc0 = nn.Linear(256 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.fc0(x)


# %%
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.LayerNorm):
        # PyTorch's default already does weight=1, bias=0; kept explicit for clarity.
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# %%
model = ExpandingHead()
print("fc0 before conversion:", model.fc0)

with torch.no_grad():
    out = model(torch.randn(2, 3, 32, 32))
print("output shape:", tuple(out.shape))

backbone_params = sum(p.numel() for p in model.backbone.parameters())
head_params = sum(p.numel() for p in model.fc0.parameters())
print(f"backbone params: {backbone_params:,}")
print(f"fc0 params:      {head_params:,}")
print(f"total:           {backbone_params + head_params:,}")

model.apply(init_weights)
#model = convert_to_scaled_net(model, layers_to_convert=[model.fc0], device=device)
print(
    "fc0 after conversion:", type(model.fc0).__name__,
    "in_features=", model.fc0.in_features,
    "out_features=", model.fc0.out_features,
)

# %%
hyperparams = {
    "num_epochs": 50,
    "batch_size": BATCH_SIZE,
    "edge_importance_metric": AbsGradientEdgeMetric(nn.CrossEntropyLoss()),
    "edge_score_aggregation": "variance",
    "expansion_top_k": {},
    "pruning_thresholds": {},#"fc0": 1},
    "plateau_threshold": 0.01,
    "min_epochs_between_expansions": 1_000_000,
    "plateau_window_size": 5,
    "learning_rate": 0.1,
    "prune_after_epochs": 4,
    "task_type": "classification",
    "start_fully_connected": False,
    "max_new_edges_per_expansion": 1_000,
    "weight_decay": 1e-4,
    "momentum": 0.9,
    "diff_threshold": 0.05,
    "t_max": 50,
    "convex_lambda": 1.0,
    "convex_every_k": 1,
    "convex_subbatch_frac": 1.0,
    "convex_subsample_k": 64,
    "cluster_target_distance": 1.0,
    # "convex"      -> one-sided convexity penalty (relu(violation)²)
    # "taylor"      -> symmetric Taylor-recon error (no relu)
    # "consistency" -> two dropout-view gradient consistency penalty
    "reg_kind": "taylor",
    # Measure-only: compute the convex reg every batch for monitoring
    # (C-Reg, T-Total, correlations) but DO NOT add it to the loss. Drops the
    # second-order backward, so per-step cost ≈ baseline.
    "convex_measure_only": False,
    "grad_clip_norm": 1.0,
    "corr_window_short": 10,
    "corr_window_long": 50,
    # Cycle-boundary blockers — switch on for a clean baseline run regardless of
    # t_max / expansion config. Each is independent; setting block_lr_scheduler_reset
    # also implicitly blocks the other two by raising t_max above num_epochs.
    "block_weights_reinit": True,
    "block_optimizer_reinit": True,
    # Leave LR scheduler reset ENABLED so train.py builds a fresh CosineAnnealingLR
    # at every cycle boundary. This gives proper SGDR warm restarts: LR descends
    # base_lr → 0 over each t_max window, hitting 0 at every multiple of t_max,
    # then jumps back to base_lr for the next cycle. Setting True here keeps the
    # same scheduler across boundaries — PyTorch's CosineAnnealingLR has a
    # wraparound bug at last_epoch=T_max+1 that makes LR climb instead of stay 0.
    "block_lr_scheduler_reset": False,
}

# Sweep override: SWEEP_LAMBDA env var supersedes the hardcoded convex_lambda.
# Goes through the loss path so the reg actually shapes training (matches the
# per-λ runs in the table). Set SWEEP_MEASURE_ONLY=1 to override.
if "SWEEP_LAMBDA" in os.environ:
    hyperparams["convex_lambda"] = float(os.environ["SWEEP_LAMBDA"])
    hyperparams["convex_measure_only"] = bool(int(os.environ.get("SWEEP_MEASURE_ONLY", "0")))


# %%
def optimizer_factory(model_instance, current_lr=None, base_lr=None):
    if base_lr is None:
        base_lr = hyperparams["learning_rate"]
    if current_lr is None:
        current_lr = base_lr

#    new_optimizer = optim.Adam(
#        model_instance.parameters(),
#        lr=current_lr,
#        weight_decay=hyperparams["weight_decay"],
#    )
    new_optimizer = optim.SGD(
        model_instance.parameters(),
        lr=current_lr,
        weight_decay=hyperparams["weight_decay"],
        momentum=hyperparams["momentum"],
    )
    for group in new_optimizer.param_groups:
        group["initial_lr"] = base_lr
    return new_optimizer


# %%
runs_dir = Path("runs")
runs_dir.mkdir(exist_ok=True)
stamp = time.strftime("%Y%m%d_%H%M%S")

_orig_expand_model = ssonn_train.expand_model
_expansion_counter = {"i": 0}
criterion = nn.CrossEntropyLoss()

# Shared state used by the LR-scheduler-reset blocker: we capture per-group LRs
# at the start of the boundary block (inside _patched_expand_model) and restore
# them when the patched CosineAnnealingLR fires a few lines later.
_lr_block_state = {
    "active_optimizer": None,   # set by _convex_train_one_epoch each step
    "saved_lrs": None,          # captured pre-boundary
    "first_sched_call": True,   # the very first sched build at train start is real
    "current_sched": None,      # remembered to return on subsequent calls
}


def _patched_expand_model(model, edge_finder, thresholds):
    """Snapshot model state, val metrics, and edge metric before expansion."""
    i = _expansion_counter["i"]
    _expansion_counter["i"] += 1
    epoch_estimate = (i + 1) * hyperparams["t_max"]

    # Save pre-boundary LRs so the blocker can restore them after train.py's
    # unconditional `group['lr'] = base_lr` + `sched = CosineAnnealingLR(...)`.
    if hyperparams.get("block_lr_scheduler_reset", False):
        opt = _lr_block_state["active_optimizer"]
        if opt is not None:
            _lr_block_state["saved_lrs"] = [g["lr"] for g in opt.param_groups]

    try:
        val_loss, val_acc = eval_one_epoch(
            model, criterion, val_loader, hyperparams["task_type"], device
        )
    except Exception as e:
        print(f"[SNAPSHOT] eval_one_epoch failed: {e}")
        val_loss, val_acc = None, None

    edge_metric_dump = {}
    for layer_name, threshold in thresholds.items():
        if not hasattr(model, layer_name):
            continue
        layer = getattr(model, layer_name)
        if not hasattr(layer, "sub_linears"):
            continue
        try:
            metric = edge_finder.calculate_edge_metric_for_dataloader(
                model=model, layer=layer, to_normalise=True, embed=False
            )
        except Exception as e:
            print(f"[SNAPSHOT] edge metric for '{layer_name}' failed: {e}")
            continue
        m_np = metric.detach().cpu().numpy()
        edge_metric_dump[layer_name] = {
            "shape": tuple(metric.shape),
            "max": float(metric.max()),
            "mean": float(metric.mean()),
            "min": float(metric.min()),
            "std": float(metric.std()),
            "above_threshold": int((metric > threshold).sum().item()),
            "values_normalized": m_np,
        }

    fc0 = getattr(model, "fc0", None)
    main_shape = tuple(fc0.weight.shape) if fc0 is not None else None
    sub_layout = (
        [(int(s.weight.shape[1]), int(s.weight.shape[0])) for s in fc0.sub_linears]
        if fc0 is not None and hasattr(fc0, "sub_linears")
        else []
    )

    snapshot_path = runs_dir / f"basic_cnn_cosine_{stamp}_pre_expand_{i:02d}.pt"
    torch.save(
        {
            "phase": "pre_expansion",
            "expansion_index": i,
            "epoch_estimate": epoch_estimate,
            "model": copy.deepcopy(model).cpu(),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "fc0_main_shape": main_shape,
            "fc0_sub_layout": sub_layout,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "thresholds": dict(thresholds),
            "edge_metric": edge_metric_dump,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        snapshot_path,
    )
    size_mb = snapshot_path.stat().st_size / 1e6
    print(
        f"[SNAPSHOT] pre-expansion {i} (epoch≈{epoch_estimate}) -> {snapshot_path}  "
        f"({size_mb:.1f} MB)  val_acc={val_acc}"
    )

    return _orig_expand_model(model, edge_finder, thresholds)


ssonn_train.expand_model = _patched_expand_model


# --- Cycle-boundary blockers ----------------------------------------------
# Independent flags for baseline runs. Each blocker only takes effect if the
# `epoch % t_max == 0 and epoch != 0` branch in train_ssonn is ever entered —
# block_lr_scheduler_reset additionally makes that branch unreachable.

# (1) Block weights reinitialization: train_ssonn calls `model.apply(weights_init)`
# at the boundary. We monkey-patch ssonn_train.weights_init to a no-op so even
# if the branch fires (post-expansion), no parameters get clobbered.
if hyperparams.get("block_weights_reinit", False):
    ssonn_train.weights_init = lambda _m: None
    if IS_RANK0:
        print("blocker: weights_init -> no-op")

# (2) Block optimizer reinitialization: train_ssonn calls optimizer_factory(...)
# again at the boundary. Wrap it so subsequent calls return the existing
# optimizer (preserving Adam/SGD moments) instead of rebuilding from scratch.
if hyperparams.get("block_optimizer_reinit", False):
    _real_optimizer_factory = optimizer_factory
    _opt_state = {"first": True, "last": None}

    def optimizer_factory(model_instance, current_lr=None, base_lr=None):  # noqa: F811
        if not _opt_state["first"]:
            return _opt_state["last"]
        _opt_state["first"] = False
        opt = _real_optimizer_factory(
            model_instance, current_lr=current_lr, base_lr=base_lr,
        )
        _opt_state["last"] = opt
        return opt
    if IS_RANK0:
        print("blocker: optimizer_factory -> returns existing optimizer after first call")

# (3) Block LR scheduler reset: monkey-patch CosineAnnealingLR so the cycle-
# boundary `sched = CosineAnnealingLR(...)` call returns the already-running
# scheduler, and undo the LR reset that train.py's
# `for group in optimizer.param_groups: group['lr'] = base_lr` does just before.
# t_max is left untouched — expansion can still trigger; only the side effects
# on the LR schedule are neutralised.
if hyperparams.get("block_lr_scheduler_reset", False):
    _real_cosine = ssonn_train.CosineAnnealingLR

    def _blocking_cosine(optimizer, T_max, eta_min=0, last_epoch=-1):
        if _lr_block_state["first_sched_call"]:
            s = _real_cosine(
                optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch,
            )
            _lr_block_state["first_sched_call"] = False
            _lr_block_state["current_sched"] = s
            return s
        # Boundary reset: restore the per-group LRs we captured in
        # _patched_expand_model and hand back the existing scheduler.
        if _lr_block_state["saved_lrs"] is not None:
            for g, lr in zip(optimizer.param_groups, _lr_block_state["saved_lrs"]):
                g["lr"] = lr
        return _lr_block_state["current_sched"]

    ssonn_train.CosineAnnealingLR = _blocking_cosine
    if IS_RANK0:
        print("blocker: CosineAnnealingLR -> existing scheduler returned, LRs restored")


# %%
# --- Per-neuron convexity regularizer --------------------------------------
#
# Per-nonlinearity convexity check.
#
# For every nonlinearity (here every nn.ReLU) we treat its input/output pair as
# samples of a scalar function σ: x → f.  For each scalar position we sample K
# items from the batch and impose the pairwise convexity-style penalty
#
#     reg = Σ_{i,j} max(0, f_i + g_i·(x_i − x_j) − f_j)^2
#
# where g_i is the backprop gradient at the pre-activation, i.e. ∂L_main/∂x_i,
# captured via `torch.autograd.grad`.  `create_graph=True` keeps that signal
# differentiable so the regularizer's backward reshapes both the activations
# AND the gradient flow upstream of the nonlinearity.
#
from sklearn.metrics import accuracy_score  # noqa: E402

_relu_cache: dict = {}                # {module: {"pre": x, "post": f}}
_relu_hook_handles: list = []
_last_convex_stats: dict = {
    "reg_mean": None, "applied": 0, "total": 0, "time_s": None, "total_loss": None,
    "between_dist": None,
}

# Side channel: _taylor_regularizer stashes the between-cluster distance
# (averaged over channels & layers) here per call; the train loop accumulates.
_last_between_dist = [None]  # single-element list for mutability across module scope


def _relu_target_modules(m):
    """Target every ReLU activation AND every weight-bearing layer."""
    return [mod for mod in m.modules() if isinstance(mod, (nn.ReLU, nn.Conv2d, nn.Linear))]


def _relu_hook(module, inputs, output):
    # Capture the live input/output tensors for this layer. Both stay on the
    # autograd graph so autograd.grad can deliver ∂L/∂input later.
    _relu_cache[module] = {"pre": inputs[0], "post": output}


def _register_relu_hooks(m):
    for h in _relu_hook_handles:
        h.remove()
    _relu_hook_handles.clear()
    for mod in _relu_target_modules(m):
        _relu_hook_handles.append(mod.register_forward_hook(_relu_hook))


def _per_sample_scalar(t: torch.Tensor) -> torch.Tensor:
    """Collapse everything except the batch dim to one scalar per sample.

    Conv input/output (B, C, H, W) → (B,);   Linear (B, F) → (B,).
    Needed for weight-bearing layers because input and output have different
    channel counts and don't broadcast in the pairwise convexity formula.
    """
    if t.dim() <= 1:
        return t
    return t.mean(dim=tuple(range(1, t.dim())))


def _convex_regularizer(m, main_loss):
    relus = _relu_target_modules(m)
    if not relus:
        return main_loss.new_tensor(0.0)

    xs, fs, mods = [], [], []
    for mod in relus:
        entry = _relu_cache.get(mod)
        if entry is None or "pre" not in entry or "post" not in entry:
            continue
        pre = entry["pre"]
        # Skip tensors that aren't on the autograd graph (e.g., the first conv's
        # input is the raw image which has requires_grad=False — autograd.grad
        # would reject the whole batch otherwise).
        if not pre.requires_grad:
            continue
        xs.append(pre)
        fs.append(entry["post"])
        mods.append(mod)

    if not xs:
        return main_loss.new_tensor(0.0)

    # When `convex_measure_only=True` we never backprop through the reg, so
    # we skip the second-order graph (much cheaper).
    measure_only = bool(hyperparams.get("convex_measure_only", False))
    gs = torch.autograd.grad(
        main_loss, xs,
        create_graph=not measure_only,
        retain_graph=True,
        allow_unused=True,
    )

    K = max(2, int(hyperparams.get("convex_subsample_k", 16)))
    eps = 1e-6
    reg = main_loss.new_tensor(0.0)
    counted = 0

    for x, f, g in zip(xs, fs, gs):
        if g is None:
            continue
        B = x.shape[0]
        if B < 2:
            continue
        k = min(K, B)
        if k < B:
            idx = torch.randperm(B, device=x.device)[:k]
            x_b = x.index_select(0, idx)
            f_b = f.index_select(0, idx)
            g_b = g.index_select(0, idx)
        else:
            x_b, f_b, g_b = x, f, g

        # Collapse to per-sample scalars: Conv input/output have differing
        # channel counts, so per-(sample, channel) collapse won't broadcast.
        # The convexity check operates on layer-aggregate magnitudes.
        x_b = _per_sample_scalar(x_b)
        f_b = _per_sample_scalar(f_b)
        g_b = _per_sample_scalar(g_b)

        # Pairwise convexity penalty: shape (k, k).
        xi = x_b.unsqueeze(1)
        xj = x_b.unsqueeze(0)
        fi = f_b.unsqueeze(1)
        fj = f_b.unsqueeze(0)
        gi = g_b.unsqueeze(1)
        expr = fi + gi * (xi - xj) - fj
        penalty = torch.relu(expr).pow(2).mean()

        # (2) Per-layer normalisation: divide by detached variance of the
        # post-activation summary so layers with wildly different scales
        # (deep layers have tiny g, shallow ones have larger g/f) contribute
        # comparable, dimensionless amounts.
        scale = f_b.detach().var().clamp(min=eps)
        reg = reg + penalty / scale
        counted += 1

    return reg / max(counted, 1)


def _taylor_regularizer(m, main_loss):
    """Per-channel nearest-neighbor pull on layer inputs.

    For each target layer with input shape (B, C, ...):
        collapse spatial dims → (B, C);
        per channel, sort B samples ascending by x;
        for each interior sample i ∈ [1, B-2], the closest neighbor in sorted
        order is min(x_i - x_{i-1}, x_{i+1} - x_i); penalize the squared min-gap
        and mean over interior samples and channels.

    Edge samples (i = 0 and i = B-1) are skipped — they have only one neighbor,
    and pulling them inward would be one-sided.
    """
    targets = _relu_target_modules(m)
    if not targets:
        return main_loss.new_tensor(0.0)

    reg = main_loss.new_tensor(0.0)
    counted = 0
    between_dist_acc = 0.0
    between_dist_count = 0

    for mod in targets:
        entry = _relu_cache.get(mod)
        if entry is None or "pre" not in entry:
            continue
        pre = entry["pre"]
        if not pre.requires_grad:
            continue
        B = pre.shape[0]
        if B < 3:
            continue

        # Collapse spatial dims only: (B, C, H, W) → (B, C); (B, F) stays (B, F).
        if pre.dim() > 2:
            x_bc = pre.mean(dim=tuple(range(2, pre.dim())))
        else:
            x_bc = pre

        # Sort per-channel along the batch dim.
        sort_idx = torch.argsort(x_bc.detach(), dim=0)
        x_s = torch.gather(x_bc, 0, sort_idx)  # (B, C) sorted per channel

        # 2-cluster within-cluster sum of squares.
        # 1. Find clustering border per channel: argmax of consecutive gaps.
        # 2. Split sorted samples into clusters A = positions ≤ border,
        #    B = positions > border (every channel has both non-empty since
        #    argmax(gaps) ∈ [0, B-2] guarantees |A| ≥ 1 and |B| ≥ 1).
        # 3. Penalty = Σ_clusters Σ_samples (x − cluster_mean)². The argmax-
        #    based partition is discrete (no grad), but the means and squared
        #    distances are differentiable through x_s.
        gaps = x_s[1:] - x_s[:-1]                # (B-1, C)
        border = gaps.detach().argmax(dim=0)      # (C,), constant
        positions = torch.arange(B, device=x_bc.device).unsqueeze(1)  # (B, 1)
        A_mask = (positions <= border.unsqueeze(0)).to(x_bc.dtype)    # (B, C)
        B_mask = 1.0 - A_mask                                          # (B, C)

        n_A = A_mask.sum(dim=0).clamp(min=1)
        n_B = B_mask.sum(dim=0).clamp(min=1)
        mean_A = (x_s * A_mask).sum(dim=0) / n_A   # (C,)
        mean_B = (x_s * B_mask).sum(dim=0) / n_B   # (C,)

        diff_A = (x_s - mean_A.unsqueeze(0)) * A_mask
        diff_B = (x_s - mean_B.unsqueeze(0)) * B_mask
        # Per-cluster RMS distance to cluster center (= sqrt of variance),
        # summed across the 2 clusters. Linear distance units → directly
        # comparable to BC-Dist. clamp(min=1e-12) for safe sqrt gradient at 0.
        rms_A = (diff_A.pow(2).sum(dim=0) / n_A).clamp(min=1e-12).sqrt()  # (C,)
        rms_B = (diff_B.pow(2).sum(dim=0) / n_B).clamp(min=1e-12).sqrt()  # (C,)
        creg_layer = (rms_A + rms_B).mean()                               # scalar
        # Between-cluster distance, differentiable so it can enter the loss.
        bcd_layer = (mean_B - mean_A).abs().mean()                        # scalar

        # Ratio target: minimise C/BC directly — encourages sharper bimodal
        # structure (tighter clusters AND/OR wider centre separation).
        # Small epsilon in denominator for stability when BC-Dist is tiny.
        penalty = creg_layer / (bcd_layer + 1e-6)

        # Diagnostic stash (detached) — only used for logging BC-Dist value.
        between_dist_acc += bcd_layer.detach().item()
        between_dist_count += 1

        reg = reg + penalty
        counted += 1

    if between_dist_count > 0:
        _last_between_dist[0] = between_dist_acc / between_dist_count
    else:
        _last_between_dist[0] = None
    return reg / max(counted, 1)


def _unwrap(m):
    # Handles DDP, DP, or bare modules.
    return m.module if hasattr(m, "module") else m


def _consistency_step(model, inputs, targets, criterion, lam, measure_only):
    """Two dropout-view consistency step.

    Runs model(inputs) twice. PyTorch's dropout (and any other stochastic op)
    consumes from the default RNG, so the two forwards see different masks.
    Computes per-parameter gradients of each forward's loss, returns a
    normalized squared-distance penalty between them.

    Returns (main_loss, outputs_a, reg_value_or_None).
    """
    with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
        out_a = model(inputs)
        loss_a = criterion(out_a, targets)
    with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
        out_b = model(inputs)
        loss_b = criterion(out_b, targets)
    main_loss = 0.5 * (loss_a + loss_b)

    params = [p for p in model.parameters() if p.requires_grad]
    create_graph = (lam != 0) and (not measure_only)
    g_a = torch.autograd.grad(
        loss_a, params, create_graph=create_graph,
        retain_graph=True, allow_unused=True,
    )
    g_b = torch.autograd.grad(
        loss_b, params, create_graph=create_graph,
        retain_graph=True, allow_unused=True,
    )

    # ||g_a - g_b||² / (||g_a||² + ||g_b||² + ε) — dimensionless [0, ~1].
    # Use detached norms for the denominator so the reg only shapes the
    # numerator (avoids trivially shrinking both gradients to zero).
    num = main_loss.new_tensor(0.0)
    denom = main_loss.new_tensor(0.0)
    for a, b in zip(g_a, g_b):
        if a is None or b is None:
            continue
        num = num + ((a - b).pow(2)).sum()
        denom = denom + (a.detach().pow(2).sum() + b.detach().pow(2).sum())
    reg = num / (denom + 1e-12)
    return main_loss, out_a, reg


_epoch_counter = {"i": 0}


def _convex_train_one_epoch(model, train_loader, optimizer, criterion, device, max_norm=1.0):
    model.train()
    underlying = _unwrap(model)
    # Expose the live optimizer to _patched_expand_model so block_lr_scheduler_reset
    # can snapshot per-group LRs before train.py's boundary code overwrites them.
    _lr_block_state["active_optimizer"] = optimizer
    train_loss = 0.0
    total_loss_running = 0.0
    convex_loss_running = 0.0
    between_dist_running = 0.0
    between_dist_batches = 0
    convex_batches = 0
    all_targets, all_preds = [], []
    lam = hyperparams["convex_lambda"]
    every_k = max(1, int(hyperparams.get("convex_every_k", 1)))
    _t_epoch_start = time.time()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Ensure DistributedSampler reshuffles each epoch.
    sampler = getattr(train_loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(_epoch_counter["i"])
    _epoch_counter["i"] += 1

    reg_kind = str(hyperparams.get("reg_kind", "convex"))

    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        _relu_cache.clear()

        measure_only = bool(hyperparams.get("convex_measure_only", False))
        compute_reg = (i % every_k == 0) and (measure_only or lam != 0)

        if reg_kind == "consistency":
            # Two-view dropout consistency. Two forwards on the same inputs
            # generate different dropout masks; reg = ||g_a - g_b||² normalized.
            if compute_reg:
                main_loss, outputs, reg = _consistency_step(
                    model, inputs, targets, criterion, lam, measure_only,
                )
                convex_loss_running += reg.item()
                convex_batches += 1
            else:
                # Single forward when reg is fully off (lam=0 and not measure_only).
                with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                    outputs = model(inputs)
                    main_loss = criterion(outputs, targets)
        else:
            # Single-forward, hook-based pairwise penalty path.
            # reg_kind=="convex":    one-sided convexity penalty (relu(...)²)
            # reg_kind=="taylor":    symmetric Taylor-recon error ((f - f̂)²)
            with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
                outputs = model(inputs)
                main_loss = criterion(outputs, targets)
            if compute_reg:
                if reg_kind == "taylor":
                    reg = _taylor_regularizer(underlying, main_loss)
                    if _last_between_dist[0] is not None:
                        between_dist_running += _last_between_dist[0]
                        between_dist_batches += 1
                else:
                    reg = _convex_regularizer(underlying, main_loss)
                convex_loss_running += reg.item()
                convex_batches += 1

        if (not measure_only) and lam != 0 and compute_reg:
            total = main_loss + lam * reg
        else:
            total = main_loss
        total.backward()

        clip = float(hyperparams.get("grad_clip_norm", max_norm))
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        train_loss += main_loss.item()
        total_loss_running += total.item()
        preds = torch.argmax(outputs, dim=1)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    train_loss /= len(train_loader)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _last_convex_stats["reg_mean"] = (
        convex_loss_running / convex_batches if convex_batches > 0 else None
    )
    _last_convex_stats["applied"] = convex_batches
    _last_convex_stats["total"] = len(train_loader)
    _last_convex_stats["time_s"] = time.time() - _t_epoch_start
    _last_convex_stats["total_loss"] = total_loss_running / len(train_loader)
    _last_convex_stats["between_dist"] = (
        between_dist_running / between_dist_batches if between_dist_batches > 0 else None
    )
    return train_loss, accuracy_score(all_targets, all_preds)


_register_relu_hooks(model)
ssonn_train.train_one_epoch = _convex_train_one_epoch

# DDP wrap (per-process / per-GPU; each rank holds one full replica).
# Hooks survive intact because DDP's forward just calls self.module(...) —
# there is no replication-to-other-devices step like DataParallel does, so the
# activation cache populated during the main forward is valid for the reg.
model = model.to(device)
if IS_DISTRIBUTED:
    from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)
    if IS_RANK0:
        print(f"wrapped model in DDP (world_size={WORLD_SIZE})")
elif IS_RANK0:
    print(f"single-process, available cuda devices: {torch.cuda.device_count()}")


# --- Append convex stats to train_ssonn's epoch summary -------------------
# train_ssonn builds its "Epoch N/M | T-Loss: ..." line locally and calls
# print() directly, so we intercept print() and inject our stats only on that
# specific line — leaving every other print untouched.
import builtins  # noqa: E402
import math      # noqa: E402
import re        # noqa: E402

_orig_builtin_print = builtins.print

# History for rolling Pearson correlations between C-Reg and the standard loss
# metrics. We want a stable "is the model healthy" signal that doesn't depend
# on a single epoch's noisy reg value.
_metric_history: dict = {
    "t_loss": [], "v_loss": [], "t_acc": [], "v_acc": [], "c_reg": [],
}

# Track best validation accuracy independently of _metric_history (which only
# accumulates when C-Reg is also available).
_best_v_acc: dict = {"value": None, "epoch": None}

_epoch_line_re = re.compile(
    r"Epoch \d+/\d+\s*\|\s*"
    r"T-Loss:\s*([-\d.eE+]+)\s*\|\s*V-Loss:\s*([-\d.eE+]+)\s*\|\s*"
    r"T-Acc:\s*([-\d.eE+]+)\s*\|\s*V-Acc:\s*([-\d.eE+]+)"
)


def _pearson(x: list, y: list):
    """Pearson correlation; returns None if undefined (<2 points or zero variance)."""
    n = len(x)
    if n < 2 or n != len(y):
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return sxy / math.sqrt(sxx * syy)


def _print_with_convex(*args, **kwargs):
    if not IS_RANK0:
        return  # silence non-rank-0 processes under DDP
    if (
        args
        and isinstance(args[0], str)
        and args[0].startswith("Epoch ")
        and "T-Loss:" in args[0]
    ):
        line = args[0]
        # Strip "| Params: X/Y" segment emitted by train_ssonn before reformatting.
        line = re.sub(r"\s*\|\s*Params:\s*\d+/\d+", "", line)

        # Capture the four standard metrics from the original line and pair
        # them with this epoch's C-Reg in _metric_history.
        m = _epoch_line_re.search(line)
        creg = _last_convex_stats.get("reg_mean")
        if m and creg is not None:
            _metric_history["t_loss"].append(float(m.group(1)))
            _metric_history["v_loss"].append(float(m.group(2)))
            _metric_history["t_acc"].append(float(m.group(3)))
            _metric_history["v_acc"].append(float(m.group(4)))
            _metric_history["c_reg"].append(float(creg))

        # Track best V-Acc across the whole run, independent of C-Reg state.
        if m:
            v_acc_now = float(m.group(4))
            epoch_m = re.match(r"Epoch (\d+)/", line)
            epoch_now = int(epoch_m.group(1)) if epoch_m else None
            if _best_v_acc["value"] is None or v_acc_now > _best_v_acc["value"]:
                _best_v_acc["value"] = v_acc_now
                _best_v_acc["epoch"] = epoch_now

        # Inject T-Total right after "Epoch N/M" so it sits before T-Loss.
        if _last_convex_stats.get("total_loss") is not None:
            tt = f" | T-Total: {_last_convex_stats['total_loss']:.4f}"
            line = re.sub(r"^(Epoch \d+/\d+)", lambda m: m.group(1) + tt, line)
        creg_val = _last_convex_stats["reg_mean"]
        bcd_val = _last_convex_stats.get("between_dist")
        if creg_val is not None:
            line += f" | C-Reg: {creg_val:.4f}"
        if bcd_val is not None:
            line += f" | BC-Dist: {bcd_val:.4f}"
        if creg_val is not None and bcd_val is not None and bcd_val > 0:
            line += f" | C/BC: {creg_val / bcd_val:.4f}"
        if _last_convex_stats.get("time_s") is not None:
            line += f" | t: {_last_convex_stats['time_s']:.2f}s"
        if _best_v_acc["value"] is not None:
            tag = f"@e{_best_v_acc['epoch']}" if _best_v_acc["epoch"] is not None else ""
            line += f" | Best V-Acc: {_best_v_acc['value']:.4f} {tag}".rstrip()

        args = (line,) + args[1:]
    _orig_builtin_print(*args, **kwargs)


builtins.print = _print_with_convex

# %%
USE_COMET = False
experiment = None
if USE_COMET:
    from comet_ml import start  # noqa: E402

    experiment = start(
        api_key=os.environ.get("COMET_API_KEY"),
        workspace=os.environ.get("COMET_WORKSPACE"),
        project_name="cifar10-basic-cnn",
    )
    experiment.log_parameters(hyperparams)
    experiment.set_name("basic-cnn + ssonn-tail (cosine)")

# %%
t_train_start = time.time()
trained_model = train_ssonn(
    model,
    train_loader,
    val_loader,
    finder_loader,
    criterion,
    optimizer_factory,
    hyperparams,
    device,
    experiment,
    USE_COMET,
)
train_seconds = time.time() - t_train_start
print(f"train wall-clock: {train_seconds:.1f} s ({train_seconds/3600:.2f} h)")

ssonn_train.expand_model = _orig_expand_model

# Unwrap so post-train code (`trained_model.fc0`, `.backbone`, …) works regardless of DP.
trained_model = _unwrap(trained_model)


# %%
def evaluate(model, loader, device, criterion=None):
    model.eval()
    all_logits, all_targets = [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())
            if criterion is not None:
                total_loss += criterion(logits, y).item()
                n_batches += 1
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    preds = logits.argmax(dim=1)
    top1 = (preds == targets).float().mean().item()
    top5 = (
        logits.topk(5, dim=1).indices.eq(targets.unsqueeze(1)).any(dim=1).float().mean().item()
    )
    return {
        "loss": (total_loss / n_batches) if criterion is not None else None,
        "top1": top1,
        "top5": top5,
        "logits": logits.numpy(),
        "preds": preds.numpy(),
        "targets": targets.numpy(),
    }


if IS_RANK0:
    test_metrics = evaluate(trained_model, test_loader, device, criterion)
    acc = test_metrics["top1"]

    print(
        f"test  loss={test_metrics['loss']:.4f}"
        f"  top1={test_metrics['top1'] * 100:.2f}%"
        f"  top5={test_metrics['top5'] * 100:.2f}%"
        f"  (basic CNN backbone + SSONN tail, cosine-annealing)"
    )

# %%
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from ssonn.train.logging import get_param_counts  # noqa: E402

if IS_RANK0:
    ckpt_path = runs_dir / f"basic_cnn_cosine_{stamp}.pt"
    test_logits_path = runs_dir / f"basic_cnn_cosine_{stamp}_test_logits.npy"

    fc0 = trained_model.fc0
    sub_linears = getattr(fc0, "sub_linears", [])
    sub_layout = [(int(s.weight.shape[1]), int(s.weight.shape[0])) for s in sub_linears]
    tail_total, tail_nonzero = get_param_counts(trained_model)

    env_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    params_info = {
        "total":         sum(p.numel() for p in trained_model.parameters()),
        "trainable":     sum(p.numel() for p in trained_model.parameters() if p.requires_grad),
        "backbone":      sum(p.numel() for p in trained_model.backbone.parameters()),
        "tail_total":    tail_total,
        "tail_non_zero": tail_nonzero,
        "tail_sparsity": 1.0 - tail_nonzero / tail_total if tail_total else 0.0,
        "expansions":    len(sub_linears),
    }

    np.save(test_logits_path, test_metrics["logits"])

    torch.save(
        {
            "model": trained_model,
            "state_dict": trained_model.state_dict(),
            "fc0_main_shape": tuple(fc0.weight.shape),
            "fc0_sub_layout": sub_layout,
            "test_accuracy": acc,
            "test_top5": test_metrics["top5"],
            "test_loss": test_metrics["loss"],
            "val_accuracy": acc,
            "val_top5": test_metrics["top5"],
            "val_loss": test_metrics["loss"],
            "test_per_class": classification_report(
                test_metrics["targets"], test_metrics["preds"], output_dict=True
            ),
            "test_confusion": confusion_matrix(
                test_metrics["targets"], test_metrics["preds"]
            ).tolist(),
            "hyperparams": hyperparams,
            "seed": SEED,
            "num_epochs": hyperparams["num_epochs"],
            "train_seconds": train_seconds,
            "env": env_info,
            "params": params_info,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "test_logits_npy": test_logits_path.name,
            "val_logits_npy": test_logits_path.name,
            "expansion_snapshots": sorted(p.name for p in runs_dir.glob(
                f"basic_cnn_cosine_{stamp}_pre_expand_*.pt"
            )),
        },
        ckpt_path,
    )
    print(f"saved checkpoint  -> {ckpt_path}  ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
    print(f"saved test logits -> {test_logits_path}  ({test_logits_path.stat().st_size / 1e6:.2f} MB)")

    # %%
    if USE_COMET and experiment is not None:
        experiment.log_metric("test_accuracy", acc)
        experiment.log_other("checkpoint_path", str(ckpt_path))
        experiment.end()

if IS_DISTRIBUTED:
    dist.destroy_process_group()
