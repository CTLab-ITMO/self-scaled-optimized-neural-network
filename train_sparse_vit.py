"""
Train SparseViT (SSONN QKV) on:
  - MNIST
  - FashionMNIST
  - Cats vs Dogs (PetImages)

This is an executable alternative to running the full `experiments/ViT_experiments/vit.ipynb`.
It keeps only what is needed to reproduce the final training loop + sparse growth/prune.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from PIL import Image, ImageFile

# Allow truncated images in PetImages-like datasets
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Ensure project imports work when launched from anywhere (server/cluster/sbatch),
# without relying on an extra helper module.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ssonn.metrics.edge_finder import EdgeFinder
from ssonn.metrics.nonlinearity_metrics import AbsGradientEdgeMetric
from ssonn.model.model import ExpandingLinear
from ssonn.train.train import edge_deletion_func_new_layer, edge_replacement_func_new_layer


# -----------------------------
# Data
# -----------------------------


class PetImagesDataset(Dataset):
    """
    Minimal Cats vs Dogs dataset for the common PetImages folder:
      root/
        Cat/*.jpg
        Dog/*.jpg

    It skips unreadable images on the fly.
    """

    def __init__(self, root: str, transform=None, max_items: Optional[int] = None):
        self.root = Path(root)
        self.transform = transform

        cat_dir = self.root / "Cat"
        dog_dir = self.root / "Dog"
        if not cat_dir.exists() or not dog_dir.exists():
            raise FileNotFoundError(
                f"Expected folders {cat_dir} and {dog_dir}. "
                f"Got: root={self.root} (exists={self.root.exists()})"
            )

        items: List[Tuple[str, int]] = []
        for label_name, label in [("Cat", 0), ("Dog", 1)]:
            d = self.root / label_name
            for p in sorted(d.glob("*")):
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    items.append((str(p), label))

        if max_items is not None:
            items = items[: max_items]

        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # fallback: try next sample (keeps dataloader running)
            # NOTE: This can recurse; with heavy corruption consider pre-filtering.
            return self.__getitem__((idx + 1) % len(self.items))

        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def make_mnist_like_dataloaders(
    dataset_name: str,
    data_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
):
    assert dataset_name in {"mnist", "fashion_mnist"}

    # ViT expects 3 channels; normalize approximately.
    t_train = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    t_test = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    ds_cls = datasets.MNIST if dataset_name == "mnist" else datasets.FashionMNIST
    train_full = ds_cls(root=data_root, train=True, download=True, transform=t_train)
    test_ds = ds_cls(root=data_root, train=False, download=True, transform=t_test)

    g = torch.Generator().manual_seed(seed)
    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size], generator=g)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
    }
    return loaders


def make_catsdogs_dataloaders(
    petimages_root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_items: Optional[int] = None,
):
    # Imagenet-like normalization works fine.
    t_train = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    t_eval = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    full = PetImagesDataset(petimages_root, transform=t_train, max_items=max_items)
    # For eval loaders, we want deterministic transforms:
    full_eval = PetImagesDataset(petimages_root, transform=t_eval, max_items=max_items)

    n = len(full)
    n_test = int(n * test_ratio)
    n_val = int((n - n_test) * val_ratio)
    n_train = n - n_test - n_val

    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=g)

    # Ensure val/test use eval transforms (random_split keeps indices).
    # We re-wrap using Subset-like behavior by mapping indices into full_eval.
    train_indices = train_ds.indices  # type: ignore[attr-defined]
    val_indices = val_ds.indices  # type: ignore[attr-defined]
    test_indices = test_ds.indices  # type: ignore[attr-defined]

    train_ds = torch.utils.data.Subset(full, train_indices)
    val_ds = torch.utils.data.Subset(full_eval, val_indices)
    test_ds = torch.utils.data.Subset(full_eval, test_indices)

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "test": DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        ),
    }
    return loaders


# -----------------------------
# Model (ported from vit.ipynb)
# -----------------------------


def dense_to_sparse(dense_tensor: torch.Tensor, device="cpu") -> torch.Tensor:
    indices = dense_tensor.nonzero(as_tuple=True)
    values = dense_tensor[indices]
    indices = torch.stack(indices).to(device)
    return torch.sparse_coo_tensor(indices, values, dense_tensor.size(), device=device)


def convert_dense_to_sparse_network(model: nn.Module, layers, device="cpu") -> nn.Module:
    """
    Notebook-compatible conversion: finds modules via named_modules (works for nested).
    """
    from copy import deepcopy

    new_model = deepcopy(model)
    for name, module in model.named_modules():
        if module in layers:
            sparse_weight = dense_to_sparse(module.weight.data, device=device)
            sparse_bias = dense_to_sparse(module.bias.data, device=device)

            # Set by walking the attribute path on the copied module.
            if "." in name:
                parts = name.split(".")
                parent = new_model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], ExpandingLinear(sparse_weight, sparse_bias, device=device))
            else:
                setattr(new_model, name, ExpandingLinear(sparse_weight, sparse_bias, device=device))
    return new_model


class SimpleFCN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc0 = nn.Linear(input_size, output_size)

    def forward(self, x):
        # x: [B, N, input_size]  or [B*N, input_size]
        if x.dim() == 3:
            B, N, D = x.shape
            x = x.reshape(B * N, D)  # [B*N, D]
            x = self.fc0(x)  # [B*N, out]
            x = x.reshape(B, N, -1)  # [B, N, out]
            return x
        return self.fc0(x)


def make_sparse_qkv_module(d_model: int, device: str):
    dense_qkv = SimpleFCN(input_size=d_model, output_size=3 * d_model)
    sparse_qkv = convert_dense_to_sparse_network(dense_qkv, layers=[dense_qkv.fc0], device=device)
    return sparse_qkv


def make_dense_qkv_module(d_model: int):
    # Same API/shape as sparse QKV: [B,N,d] -> [B,N,3d]
    return SimpleFCN(input_size=d_model, output_size=3 * d_model)


class CustomMHA(nn.Module):
    def __init__(self, d_model, num_heads, qkv_module, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_module = qkv_module  # sparse QKV: [B,N,d] -> [B,N,3d]
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        qkv = self.qkv_module(x)  # [B, N, 3d]
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, Hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.out_proj(out)
        return out


class ViTEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, qkv_module, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CustomMHA(d_model, num_heads, qkv_module, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, mask)
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class SparseViT(nn.Module):
    def __init__(
        self,
        num_classes,
        img_size=224,
        patch_size=16,
        dim=768,
        depth=2,
        num_heads=8,
        channels=3,
        dropout=0.1,
        device="cpu",
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.dim = dim

        self.patch_embeddings = nn.Conv2d(
            in_channels=channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))

        self.encoder_blocks = nn.ModuleList()
        for i in range(depth):
            qkv_sparse = make_sparse_qkv_module(dim, device)
            setattr(self, f"qkv_layer_{i}", qkv_sparse)  # used by sparse expansion logic
            block = ViTEncoderBlock(dim, num_heads, qkv_sparse, dropout=dropout)
            self.encoder_blocks.append(block)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, mask=None):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, d]

        B = x.size(0)
        cls = self.class_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.position_embeddings

        for block in self.encoder_blocks:
            x = block(x, mask)

        x = x[:, 0]
        x = self.dropout(x)
        return self.head(x)


class DenseViT(nn.Module):
    """Architecturally same as SparseViT, but QKV is dense nn.Linear (inside SimpleFCN)."""

    def __init__(
        self,
        num_classes,
        img_size=224,
        patch_size=16,
        dim=768,
        depth=2,
        num_heads=8,
        channels=3,
        dropout=0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.dim = dim

        self.patch_embeddings = nn.Conv2d(
            in_channels=channels,
            out_channels=dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.class_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))

        self.encoder_blocks = nn.ModuleList()
        for i in range(depth):
            qkv_dense = make_dense_qkv_module(dim)
            setattr(self, f"qkv_layer_{i}", qkv_dense)  # for name parity in logs/configs
            block = ViTEncoderBlock(dim, num_heads, qkv_dense, dropout=dropout)
            self.encoder_blocks.append(block)

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x, mask=None):
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, d]

        B = x.size(0)
        cls = self.class_token.expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.position_embeddings

        for block in self.encoder_blocks:
            x = block(x, mask)

        x = x[:, 0]
        x = self.dropout(x)
        return self.head(x)


# -----------------------------
# Training + sparse growth/prune
# -----------------------------


def get_params_amount(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def _confusion_matrix_binary(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    # y_true/y_pred: [N] with values 0/1
    cm = torch.zeros((2, 2), dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t][p] += 1
    return cm


@torch.no_grad()
def _confusion_matrix_multiclass(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> torch.Tensor:
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t][p] += 1
    return cm


def run_epoch(dataloader, model, criterion, optimizer, device, phase: str, num_classes: int):
    assert phase in {"train", "eval"}
    if phase == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_corrects = 0
    total = 0

    all_y = []
    all_pred = []

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if phase == "train":
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(phase == "train"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            if phase == "train":
                loss.backward()
                optimizer.step()

        bs = labels.size(0)
        running_loss += loss.item() * bs
        running_corrects += torch.sum(preds == labels).item()
        total += bs

        all_y.append(labels.detach().cpu())
        all_pred.append(preds.detach().cpu())

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = running_corrects / max(total, 1)

    y = torch.cat(all_y) if all_y else torch.tensor([], dtype=torch.long)
    p = torch.cat(all_pred) if all_pred else torch.tensor([], dtype=torch.long)

    if not y.numel():
        cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    elif num_classes == 2:
        cm = _confusion_matrix_binary(y, p)
    else:
        cm = _confusion_matrix_multiclass(y, p, num_classes=num_classes)
    return epoch_loss, epoch_acc, cm


def apply_sparse_growth_and_prune(
    model,
    optimizer,
    edge_finder,
    hyperparams,
    epoch,
    val_losses,
    replace_epoch,
    non_zero_masks,
    logs,
):
    # Default debug fields
    logs.setdefault("plateau_checked", 0)
    logs.setdefault("plateau_triggered", 0)
    logs.setdefault("added_edges", 0)
    logs.setdefault("deleted_edges", logs.get("deleted_edges", 0))

    if len(val_losses) < hyperparams["plateau_window_size"]:
        return

    if epoch - replace_epoch[-1] > max(
        hyperparams["prune_after_epochs"],
        hyperparams["min_epochs_between_expansions"],
        hyperparams["plateau_window_size"],
    ):
        logs["plateau_checked"] = 1
        recent_changes = [
            abs(val_losses[i] - val_losses[i - 1])
            for i in range(-hyperparams["plateau_window_size"], 0)
        ]
        avg_change = sum(recent_changes) / hyperparams["plateau_window_size"]
        logs["plateau_avg_change"] = avg_change
        print(avg_change)
        if avg_change < hyperparams["plateau_threshold"]:
            logs["plateau_triggered"] = 1
            len_choose = 0
            for layer_name in hyperparams["expansion_thresholds"].keys():
                layer = getattr(model, layer_name)
                if hasattr(layer, "fc0") and isinstance(layer.fc0, ExpandingLinear):
                    expanding_layer = layer.fc0
                elif isinstance(layer, ExpandingLinear):
                    expanding_layer = layer
                else:
                    continue

                len_choose += edge_replacement_func_new_layer(
                    model,
                    expanding_layer,
                    optimizer,
                    hyperparams["expansion_thresholds"][layer_name],
                    edge_finder,
                    experiment=None,
                    fully_connected=hyperparams["start_fully_connected"],
                )
                non_zero_masks[layer_name] = expanding_layer.get_non_zero_params()
            logs["added_edges"] = int(len_choose)
            replace_epoch.append(epoch)

    if (
        epoch - replace_epoch[-1] == hyperparams["prune_after_epochs"]
        and replace_epoch[-1] != 0
    ):
        len_choose = 0
        for layer_name in hyperparams["pruning_thresholds"].keys():
            layer = getattr(model, layer_name)
            if hasattr(layer, "fc0") and isinstance(layer.fc0, ExpandingLinear):
                expanding_layer = layer.fc0
            elif isinstance(layer, ExpandingLinear):
                expanding_layer = layer
            else:
                continue
            len_choose += edge_deletion_func_new_layer(
                model,
                expanding_layer,
                optimizer,
                non_zero_masks[layer_name],
                hyperparams["pruning_thresholds"][layer_name],
                edge_finder,
            )
        logs["deleted_edges"] = len_choose


def train_sparse_vit(
    model,
    dataloaders,
    criterion,
    optimizer,
    hyperparams,
    device,
    num_epochs,
    run_dir: Path,
    num_classes: int,
    enable_expansion: bool,
):
    model = model.to(device)

    val_loader_for_edges = dataloaders.get("val", dataloaders["train"])
    edge_finder = EdgeFinder(
        hyperparams["edge_importance_metric"],
        val_loader_for_edges,
        device=device,
        aggregation_mode=hyperparams["edge_score_aggregation"],
        max_to_replace=hyperparams["max_new_edges_per_expansion"],
    )

    phases = ["train", "val", "test"] if "val" in dataloaders else ["train", "test"]

    saved = {p: {"loss": [], "acc": []} for p in phases}
    val_losses = []
    non_zero_masks = {}
    replace_epoch = [0]

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "phase",
                "loss",
                "acc",
                "p00",
                "p01",
                "p10",
                "p11",
                "cm_json",
                "params",
                "plateau_avg_change",
                "deleted_edges",
            ],
        )
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print("=" * 80)
        print(f"Epoch {epoch}/{num_epochs}")

        epoch_logs: Dict[str, float] = {}

        for phase in phases:
            dl = dataloaders[phase]
            phase_mode = "train" if phase == "train" else "eval"
            loss, acc, cm = run_epoch(dl, model, criterion, optimizer, device, phase_mode, num_classes=num_classes)

            saved[phase]["loss"].append(loss)
            saved[phase]["acc"].append(acc)

            print(f"{phase}: loss={loss:.6f} acc={acc:.4f} cm={cm.tolist()}")

            # plateau tracking: prefer val, else test
            if ("val" in dataloaders and phase == "val") or ("val" not in dataloaders and phase == "test"):
                val_losses.append(loss)

            params_amount = get_params_amount(model)
            with metrics_path.open("a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "phase",
                        "loss",
                        "acc",
                        "p00",
                        "p01",
                        "p10",
                        "p11",
                        "cm_json",
                        "params",
                        "plateau_avg_change",
                        "deleted_edges",
                    ],
                )
                cm_json = json.dumps(cm.tolist())
                writer.writerow(
                    {
                        "epoch": epoch,
                        "phase": phase,
                        "loss": loss,
                        "acc": acc,
                        "p00": int(cm[0, 0]) if num_classes == 2 else "",
                        "p01": int(cm[0, 1]) if num_classes == 2 else "",
                        "p10": int(cm[1, 0]) if num_classes == 2 else "",
                        "p11": int(cm[1, 1]) if num_classes == 2 else "",
                        "cm_json": cm_json,
                        "params": params_amount,
                        "plateau_avg_change": epoch_logs.get("plateau_avg_change", ""),
                        "deleted_edges": epoch_logs.get("deleted_edges", ""),
                    }
                )

        # growth/prune once per epoch (after all phases)
        if enable_expansion and len(val_losses) >= hyperparams["plateau_window_size"]:
            apply_sparse_growth_and_prune(
                model=model,
                optimizer=optimizer,
                edge_finder=edge_finder,
                hyperparams=hyperparams,
                epoch=epoch,
                val_losses=val_losses,
                replace_epoch=replace_epoch,
                non_zero_masks=non_zero_masks,
                logs=epoch_logs,
            )
            # Human-readable debug line (why expansion did/didn't happen)
            if epoch_logs.get("plateau_checked", 0):
                avg = epoch_logs.get("plateau_avg_change", None)
                thr = hyperparams.get("plateau_threshold", None)
                trig = int(epoch_logs.get("plateau_triggered", 0))
                added = int(epoch_logs.get("added_edges", 0))
                deleted = int(epoch_logs.get("deleted_edges", 0) or 0)
                print(
                    f"[plateau] avg_change={avg:.6f} thr={thr} triggered={trig} "
                    f"added_edges={added} deleted_edges={deleted}"
                )
            else:
                # Not checked because of min-epochs-between-expansions / prune window
                print(
                    "[plateau] skipped (min_epochs_between_expansions/prune/window constraint)"
                )
            print("p", get_params_amount(model))
        elif enable_expansion:
            # Not enough val points yet
            need = hyperparams["plateau_window_size"]
            print(f"[plateau] waiting: val_losses={len(val_losses)}/{need}")

    (run_dir / "hyperparams.json").write_text(json.dumps(hyperparams, indent=2))
    return saved


def train_dense_vit(
    model,
    dataloaders,
    criterion,
    optimizer,
    device,
    num_epochs,
    run_dir: Path,
    num_classes: int,
):
    model = model.to(device)
    phases = ["train", "val", "test"] if "val" in dataloaders else ["train", "test"]

    saved = {p: {"loss": [], "acc": []} for p in phases}

    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "phase",
                "loss",
                "acc",
                "p00",
                "p01",
                "p10",
                "p11",
                "cm_json",
                "params",
            ],
        )
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print("=" * 80)
        print(f"[DENSE] Epoch {epoch}/{num_epochs}")

        for phase in phases:
            dl = dataloaders[phase]
            phase_mode = "train" if phase == "train" else "eval"
            loss, acc, cm = run_epoch(dl, model, criterion, optimizer, device, phase_mode, num_classes=num_classes)

            saved[phase]["loss"].append(loss)
            saved[phase]["acc"].append(acc)

            print(f"{phase}: loss={loss:.6f} acc={acc:.4f} cm={cm.tolist()}")

            params_amount = get_params_amount(model)
            with metrics_path.open("a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "epoch",
                        "phase",
                        "loss",
                        "acc",
                        "p00",
                        "p01",
                        "p10",
                        "p11",
                        "cm_json",
                        "params",
                    ],
                )
                writer.writerow(
                    {
                        "epoch": epoch,
                        "phase": phase,
                        "loss": loss,
                        "acc": acc,
                        "p00": int(cm[0, 0]) if num_classes == 2 else "",
                        "p01": int(cm[0, 1]) if num_classes == 2 else "",
                        "p10": int(cm[1, 0]) if num_classes == 2 else "",
                        "p11": int(cm[1, 1]) if num_classes == 2 else "",
                        "cm_json": json.dumps(cm.tolist()),
                        "params": params_amount,
                    }
                )

    return saved


# -----------------------------
# Presets
# -----------------------------


@dataclass
class Preset:
    num_classes: int
    img_size: int
    patch_size: int
    dim: int
    depth: int
    num_heads: int
    dropout: float
    lr: float
    batch_size: int
    num_epochs: int


PRESETS: Dict[str, Preset] = {
    "mnist_small": Preset(
        num_classes=10,
        img_size=32,
        patch_size=4,
        dim=128,
        depth=2,
        num_heads=4,
        dropout=0.1,
        lr=1e-4,
        batch_size=64,
        num_epochs=10,
    ),
    "fashion_small": Preset(
        num_classes=10,
        img_size=32,
        patch_size=4,
        dim=128,
        depth=2,
        num_heads=4,
        dropout=0.1,
        lr=1e-4,
        batch_size=64,
        num_epochs=10,
    ),
    "catsdogs": Preset(
        num_classes=2,
        img_size=224,
        patch_size=16,
        dim=256,
        depth=2,
        num_heads=4,
        dropout=0.1,
        lr=1e-4,
        batch_size=32,
        num_epochs=50,
    ),
}


def build_hyperparams(depth: int, max_new_edges: int, start_fully_connected: bool):
    expansion_thresholds = {f"qkv_layer_{i}": 0.12 for i in range(depth)}
    pruning_thresholds = {f"qkv_layer_{i}": 0.06 for i in range(depth)}
    return {
        "task_type": "classification",
        "edge_importance_metric": AbsGradientEdgeMetric(nn.CrossEntropyLoss()),
        "edge_score_aggregation": "mean",
        "max_new_edges_per_expansion": max_new_edges,
        "prune_after_epochs": 3,
        "min_epochs_between_expansions": 2,
        "plateau_window_size": 3,
        "plateau_threshold": 0.001,
        "start_fully_connected": start_fully_connected,
        "expansion_thresholds": expansion_thresholds,
        "pruning_thresholds": pruning_thresholds,
    }


def build_hyperparams_from_args(depth: int, args) -> dict:
    """
    Create hyperparams with CLI overrides.
    """
    expansion_thresholds = {f"qkv_layer_{i}": float(args.expansion_threshold) for i in range(depth)}
    pruning_thresholds = {f"qkv_layer_{i}": float(args.pruning_threshold) for i in range(depth)}
    return {
        "task_type": "classification",
        "edge_importance_metric": AbsGradientEdgeMetric(nn.CrossEntropyLoss()),
        "edge_score_aggregation": str(args.edge_score_aggregation),
        "max_new_edges_per_expansion": int(args.max_new_edges),
        "prune_after_epochs": int(args.prune_after_epochs),
        "min_epochs_between_expansions": int(args.min_epochs_between_expansions),
        "plateau_window_size": int(args.plateau_window_size),
        "plateau_threshold": float(args.plateau_threshold),
        "start_fully_connected": False,  # запрещено
        "expansion_thresholds": expansion_thresholds,
        "pruning_thresholds": pruning_thresholds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mnist", "fashion_mnist", "catsdogs"], required=True)
    ap.add_argument("--model", choices=["sparse", "dense"], default="sparse")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default=None)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--petimages-root", default=None)
    ap.add_argument("--run-dir", default="./runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--max-items", type=int, default=None)

    # We intentionally do NOT support start_fully_connected=True here (it was unstable in our setup).
    ap.add_argument("--start-fully-connected", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--max-new-edges", type=int, default=100)
    ap.add_argument("--no-expansion", action="store_true", help="Отключить growth/prune (чистый запуск без расширения)")

    # Sparse growth/prune knobs (important for noisy datasets like CatsDogs)
    ap.add_argument("--plateau-threshold", type=float, default=None, help="Порог плато по val-loss для запуска расширения")
    ap.add_argument("--plateau-window-size", type=int, default=None, help="Окно (эпохи) для оценки плато")
    ap.add_argument("--min-epochs-between-expansions", type=int, default=None, help="Мин. эпох между расширениями")
    ap.add_argument("--prune-after-epochs", type=int, default=None, help="Через сколько эпох после расширения делать pruning")
    ap.add_argument("--expansion-threshold", type=float, default=None, help="Порог метрики ребра для добавления (один на все QKV)")
    ap.add_argument("--pruning-threshold", type=float, default=None, help="Порог метрики ребра для удаления (один на все QKV)")
    ap.add_argument("--edge-score-aggregation", choices=["mean", "sum", "max"], default="mean")

    # override model params
    ap.add_argument("--img-size", type=int, default=None)
    ap.add_argument("--patch-size", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--num-heads", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)

    args = ap.parse_args()
    if args.start_fully_connected:
        raise ValueError(
            "start_fully_connected=True запрещён в этом проекте (из-за нестабильностей). "
            "Убери флаг --start-fully-connected."
        )

    if args.preset is None:
        if args.dataset == "mnist":
            preset = PRESETS["mnist_small"]
        elif args.dataset == "fashion_mnist":
            preset = PRESETS["fashion_small"]
        else:
            preset = PRESETS["catsdogs"]
    else:
        preset = PRESETS[args.preset]

    img_size = args.img_size or preset.img_size
    patch_size = args.patch_size or preset.patch_size
    dim = args.dim or preset.dim
    depth = args.depth or preset.depth
    num_heads = args.num_heads or preset.num_heads
    dropout = args.dropout if args.dropout is not None else preset.dropout
    batch_size = args.batch_size or preset.batch_size
    lr = args.lr or preset.lr
    epochs = args.epochs or preset.num_epochs

    if img_size % patch_size != 0:
        raise ValueError(f"img_size must be divisible by patch_size. Got {img_size} and {patch_size}")
    if dim % num_heads != 0:
        raise ValueError(f"dim must be divisible by num_heads. Got {dim} and {num_heads}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(8, os.cpu_count() or 2)

    run_name = f"{args.dataset}_img{img_size}_p{patch_size}_d{dim}_L{depth}_H{num_heads}_{int(time.time())}"
    run_dir = Path(args.run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("Run dir:", run_dir)
    print("Device:", device)

    if args.dataset in {"mnist", "fashion_mnist"}:
        dataloaders = make_mnist_like_dataloaders(
            dataset_name=args.dataset,
            data_root=args.data_root,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        num_classes = 10
    else:
        if args.petimages_root is None:
            raise ValueError("--petimages-root is required for catsdogs")
        dataloaders = make_catsdogs_dataloaders(
            petimages_root=args.petimages_root,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            max_items=args.max_items,
        )
        num_classes = 2

    if args.model == "dense":
        model = DenseViT(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            channels=3,
            dropout=dropout,
        ).to(device)
    else:
        model = SparseViT(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            channels=3,
            dropout=dropout,
            device=str(device),
        ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # dataset-specific defaults for growth/prune (CatsDogs loss is noisy, needs higher plateau_threshold)
    if args.dataset == "catsdogs":
        default_plateau_threshold = 0.02
        default_plateau_window = 2
        default_min_between = 6
        default_prune_after = 2
        default_expansion_thr = 0.08
        default_pruning_thr = 0.04
    else:
        default_plateau_threshold = 0.002
        default_plateau_window = 3
        default_min_between = 2
        default_prune_after = 3
        default_expansion_thr = 0.12
        default_pruning_thr = 0.06

    # fill None with defaults
    if args.plateau_threshold is None:
        args.plateau_threshold = default_plateau_threshold
    if args.plateau_window_size is None:
        args.plateau_window_size = default_plateau_window
    if args.min_epochs_between_expansions is None:
        args.min_epochs_between_expansions = default_min_between
    if args.prune_after_epochs is None:
        args.prune_after_epochs = default_prune_after
    if args.expansion_threshold is None:
        args.expansion_threshold = default_expansion_thr
    if args.pruning_threshold is None:
        args.pruning_threshold = default_pruning_thr

    hyperparams = build_hyperparams_from_args(depth=depth, args=args)

    print("Model params:", get_params_amount(model))
    print("Starting training...")
    enable_expansion = (args.model == "sparse") and (not args.no_expansion) and (args.max_new_edges > 0)

    if args.model == "dense":
        train_dense_vit(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=epochs,
            run_dir=run_dir,
            num_classes=num_classes,
        )
    else:
        train_sparse_vit(
            model=model,
            dataloaders=dataloaders,
            criterion=criterion,
            optimizer=optimizer,
            hyperparams=hyperparams,
            device=device,
            num_epochs=epochs,
            run_dir=run_dir,
            num_classes=num_classes,
            enable_expansion=enable_expansion,
        )


if __name__ == "__main__":
    main()


