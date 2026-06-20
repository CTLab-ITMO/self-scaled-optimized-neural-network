#!/usr/bin/env python3
"""Standalone CIFAR training script converted from cifar10.ipynb."""

from __future__ import annotations

import argparse
import os
import random
import warnings

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torchvision.models as models
from sklearn.metrics import accuracy_score, mean_squared_error
from tqdm import tqdm

from ssonn.metrics.nonlinearity_metrics import AbsGradientEdgeMetric
import ssonn.train.train as ssonn_train
from ssonn.train.train import convert_to_scaled_net, train_ssonn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SSONN CIFAR classifier.")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--model-type", choices=("vgg19", "resnext50"), default="vgg19")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-comet", action="store_true")
    return parser.parse_args()


def set_seed(seed: int, deterministic: bool) -> torch.Generator:
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            torch.set_deterministic_debug_mode(True)
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            torch.use_deterministic_algorithms(False)
            os.environ.pop("CUDA_LAUNCH_BLOCKING", None)

    return generator


def dataset_config(dataset_name: str):
    if dataset_name == "cifar100":
        return {
            "mean": (0.5071, 0.4867, 0.4408),
            "std": (0.2675, 0.2565, 0.2761),
            "dataset_class": datasets.CIFAR100,
            "num_classes": 100,
        }

    return {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "dataset_class": datasets.CIFAR10,
        "num_classes": 10,
    }


def make_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    eval_batch_size: int,
    generator: torch.Generator,
    num_workers: int,
    pin_memory: bool,
):
    config = dataset_config(dataset_name)
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            v2.RandomErasing(p=0.5, scale=(0.1, 0.1), ratio=(1.0, 1.0), value=0),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(config["mean"], config["std"]),
        ]
    )

    dataset_class = config["dataset_class"]
    train_dataset = dataset_class(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = dataset_class(root=data_dir, train=False, download=True, transform=transform_test)

    common_loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    return (
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            **common_loader_kwargs,
        ),
        DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            **common_loader_kwargs,
        ),
        DataLoader(
            train_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            **common_loader_kwargs,
        ),
        config["num_classes"],
    )


class ExpandingHead(nn.Module):
    def __init__(self, model_type: str = "vgg19", num_classes: int = 10):
        super().__init__()

        if model_type == "vgg19":
            backbone = models.vgg19_bn(weights=None)
            self.backbone = backbone.features
            in_features = 512
        else:
            backbone = models.resnext50_32x4d(weights=None)
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            in_features = 2048

        self.fc0 = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.fc0(x)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


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


def make_hyperparams(model_type: str, dataset_name: str, batch_size: int) -> dict:
    if model_type == "vgg19" and dataset_name == "cifar10":
        return {
            "num_epochs": 500,
            "batch_size": batch_size,
            "edge_importance_metric": AbsGradientEdgeMetric(nn.CrossEntropyLoss()),
            "edge_score_aggregation": "variance",
            "expansion_thresholds": {"fc0": 1e-7},
            "pruning_thresholds": {"fc0": 1},
            "plateau_threshold": 0.01,
            "min_epochs_between_expansions": 1_000_000,
            "plateau_window_size": 5,
            "learning_rate": 0.005,
            "prune_after_epochs": 4,
            "task_type": "classification",
            "start_fully_connected": False,
            "max_new_edges_per_expansion": 10_000,
            "weight_decay": 0,
            "diff_threshold": 0.05,
            "t_max": 100,
        }

    if model_type == "vgg19":
        return {
            "num_epochs": 700,
            "batch_size": batch_size,
            "edge_importance_metric": AbsGradientEdgeMetric(nn.CrossEntropyLoss()),
            "edge_score_aggregation": "variance",
            "expansion_thresholds": {"fc0": 1e-5},
            "pruning_thresholds": {"fc0": 1},
            "plateau_threshold": 0.01,
            "min_epochs_between_expansions": 1_000_000,
            "plateau_window_size": 5,
            "learning_rate": 0.005,
            "prune_after_epochs": 4,
            "task_type": "classification",
            "start_fully_connected": False,
            "max_new_edges_per_expansion": 4000,
            "weight_decay": 0.0005,
            "diff_threshold": 0.05,
            "t_max": 100,
        }

    return {
        "num_epochs": 700,
        "batch_size": batch_size,
        "edge_importance_metric": AbsGradientEdgeMetric(nn.CrossEntropyLoss()),
        "edge_score_aggregation": "variance",
        "expansion_thresholds": {"fc0": 1e-5},
        "pruning_thresholds": {"fc0": 1},
        "plateau_threshold": 0.01,
        "min_epochs_between_expansions": 1_000_000,
        "plateau_window_size": 5,
        "learning_rate": 0.1,
        "prune_after_epochs": 4,
        "task_type": "classification",
        "start_fully_connected": False,
        "max_new_edges_per_expansion": 4000,
        "weight_decay": 0.0005,
        "diff_threshold": 0.05,
        "t_max": 100,
    }


def install_amp_epoch_functions(device_type: str) -> None:
    enabled = device_type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=enabled)

    def amp_train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        max_norm: float = 1.0,
    ):
        model.train()
        train_loss = 0.0
        all_targets = []
        all_preds = []
        pbar = tqdm(train_loader, desc="Train", leave=False)

        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=enabled):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            preds = torch.argmax(outputs.detach(), dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return train_loss / len(train_loader), accuracy_score(all_targets, all_preds)

    def amp_eval_one_epoch(model, criterion, val_loader, task_type, device):
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", enabled=enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

                if task_type == "classification":
                    preds = torch.argmax(outputs, dim=1)
                else:
                    preds = outputs
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        if task_type == "classification":
            metric = accuracy_score(all_targets, all_preds)
        else:
            metric = mean_squared_error(all_targets, all_preds)
        return val_loss / len(val_loader), metric

    ssonn_train.train_one_epoch = amp_train_one_epoch
    ssonn_train.eval_one_epoch = amp_eval_one_epoch


def maybe_start_comet(use_comet: bool, hyperparams: dict, model_type: str, dataset_name: str, seed: int):
    if not use_comet:
        return None

    from comet_ml import start

    experiment = start(
        api_key=os.environ.get("COMET_API_KEY"),
        workspace=os.environ.get("COMET_WORKSPACE"),
        project_name="ssonn-experiments",
    )
    experiment.log_parameters(hyperparams)
    experiment.set_name(f"{model_type}_{dataset_name}_seed_{seed}")
    return experiment


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    generator = set_seed(args.seed, args.deterministic)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if args.amp:
        install_amp_epoch_functions(device.type)

    train_loader, val_loader, finder_loader, num_classes = make_loaders(
        args.dataset,
        args.data_dir,
        args.batch_size,
        args.eval_batch_size,
        generator,
        args.num_workers,
        device.type == "cuda",
    )

    model = ExpandingHead(model_type=args.model_type, num_classes=num_classes)
    model.apply(init_weights)
    model = convert_to_scaled_net(model, layers_to_convert=[model.fc0], device=device)
    if args.compile:
        model = torch.compile(model)

    hyperparams = make_hyperparams(args.model_type, args.dataset, args.batch_size)
    if args.epochs is not None:
        hyperparams["num_epochs"] = args.epochs

    criterion = nn.CrossEntropyLoss()

    def optimizer_factory(model_instance, current_lr=None, base_lr=None):
        if base_lr is None:
            base_lr = hyperparams["learning_rate"]
        if current_lr is None:
            current_lr = base_lr

        if args.model_type == "vgg19":
            new_optimizer = optim.Adam(
                model_instance.parameters(),
                lr=current_lr,
                weight_decay=hyperparams["weight_decay"],
            )
        else:
            new_optimizer = optim.SGD(
                model_instance.parameters(),
                lr=current_lr,
                weight_decay=hyperparams["weight_decay"],
                momentum=0.9,
            )

        for group in new_optimizer.param_groups:
            group["initial_lr"] = base_lr
        return new_optimizer

    experiment = maybe_start_comet(
        args.use_comet, hyperparams, args.model_type, args.dataset, args.seed
    )
    try:
        train_ssonn(
            model,
            train_loader,
            val_loader,
            finder_loader,
            criterion,
            optimizer_factory,
            hyperparams,
            device,
            experiment,
            args.use_comet,
        )
    finally:
        if experiment is not None:
            experiment.end()


if __name__ == "__main__":
    main()
