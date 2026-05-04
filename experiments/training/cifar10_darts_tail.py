# CIFAR-10: SSONN tail (cosine-annealing variant) on DARTS-V2 backbone.
# Run from experiments/training/ of the cosine-annealing SSONN package.

# %%
import copy
import os
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

SEED = 0
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
g = torch.Generator()
g.manual_seed(0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
print("device:", device)

# %%
BACKBONE_PATH = os.path.abspath(
    os.path.join(os.getcwd(), "..", "..", "..", "darts-master", "cnn")
)
print("DARTS cnn path:", BACKBONE_PATH)
assert os.path.isfile(os.path.join(BACKBONE_PATH, "model.py")), (
    f"darts-master/cnn/model.py not found at {BACKBONE_PATH}"
)
if BACKBONE_PATH not in sys.path:
    sys.path.insert(0, BACKBONE_PATH)

import genotypes as darts_genotypes  # noqa: E402
from model import NetworkCIFAR  # noqa: E402

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
val_dataset = test_dataset   # test set used as val target each epoch

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
finder_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)


# %%
class ExpandingHead(nn.Module):
    def __init__(self, genotype, init_channels: int = 36, layers: int = 20, num_classes: int = 10):
        super().__init__()
        backbone = NetworkCIFAR(
            C=init_channels,
            num_classes=num_classes,
            layers=layers,
            auxiliary=False,
            genotype=genotype,
        )
        backbone.drop_path_prob = 0.0
        C_prev = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.fc0 = nn.Linear(C_prev, num_classes)

    def forward(self, x):
        feats, _ = self.backbone(x)
        return self.fc0(feats)


# %%
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# %%
model = ExpandingHead(genotype=darts_genotypes.DARTS_V2)
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
model = convert_to_scaled_net(model, layers_to_convert=[model.fc0], device=device)
print("fc0 after conversion:", type(model.fc0).__name__,
      "in_features=", model.fc0.in_features,
      "out_features=", model.fc0.out_features)

# %%
hyperparams = {
    "num_epochs": 500,
    "batch_size": BATCH_SIZE,
    "edge_importance_metric": AbsGradientEdgeMetric(nn.CrossEntropyLoss()),
    "edge_score_aggregation": "variance",
    "expansion_thresholds": {"fc0": 0.0000001},
    "pruning_thresholds": {"fc0": 1},
    "plateau_threshold": 0.01,
    "min_epochs_between_expansions": 1_000_000,
    "plateau_window_size": 5,
    "learning_rate": 1e-2,
    "prune_after_epochs": 4,
    "task_type": "classification",
    "start_fully_connected": False,
    "max_new_edges_per_expansion": 10_000,
    "weight_decay": 0,
    "diff_threshold": 0.05,
    "t_max": 100,
}


# %%
def optimizer_factory(model_instance, current_lr=None, base_lr=None):
    if base_lr is None:
        base_lr = hyperparams["learning_rate"]
    if current_lr is None:
        current_lr = base_lr

    new_optimizer = optim.Adam(
        model_instance.parameters(),
        lr=current_lr,
        weight_decay=hyperparams["weight_decay"],
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


def _patched_expand_model(model, edge_finder, thresholds):
    """Snapshot model state, val metrics, and edge metric before expansion."""
    i = _expansion_counter["i"]
    _expansion_counter["i"] += 1
    epoch_estimate = hyperparams["t_max"] + 2 * hyperparams["t_max"] * i

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

    snapshot_path = runs_dir / f"darts_tail_cosine_{stamp}_pre_expand_{i:02d}.pt"
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

# %%
USE_COMET = False
experiment = None
if USE_COMET:
    from comet_ml import start  # noqa: E402

    experiment = start(
        api_key=os.environ.get("COMET_API_KEY"),
        workspace=os.environ.get("COMET_WORKSPACE"),
        project_name="cifar10-darts-tail-cosine",
    )
    experiment.log_parameters(hyperparams)
    experiment.set_name("darts-v2 + ssonn-tail (cosine)")

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


test_metrics = evaluate(trained_model, test_loader, device, criterion)
acc = test_metrics["top1"]

print(
    f"test  loss={test_metrics['loss']:.4f}"
    f"  top1={test_metrics['top1'] * 100:.2f}%"
    f"  top5={test_metrics['top5'] * 100:.2f}%"
    f"  (DARTS-V2 backbone + SSONN tail, cosine-annealing)"
)

# %%
import platform  # noqa: E402

from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from ssonn.train.logging import get_param_counts  # noqa: E402

ckpt_path = runs_dir / f"darts_tail_cosine_{stamp}.pt"
test_logits_path = runs_dir / f"darts_tail_cosine_{stamp}_test_logits.npy"

fc0 = trained_model.fc0
sub_layout = [(int(s.weight.shape[1]), int(s.weight.shape[0])) for s in fc0.sub_linears]
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
    "expansions":    len(fc0.sub_linears),
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
        "genotype": darts_genotypes.DARTS_V2,
        "seed": SEED,
        "num_epochs": hyperparams["num_epochs"],
        "train_seconds": train_seconds,
        "env": env_info,
        "params": params_info,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_logits_npy": test_logits_path.name,
        "val_logits_npy": test_logits_path.name,
        "expansion_snapshots": sorted(p.name for p in runs_dir.glob(
            f"darts_tail_cosine_{stamp}_pre_expand_*.pt"
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
