"""
Эксперимент 2: Sanity-бейзлайн Directional Grid-8 GNN на MNIST
- 20 эпох
- Acc = 91.98% 0.19 п. п.
- Результаты в results/directional_grid8.csv
"""

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_mnist_dataloaders
from src.models import SimpleGNN, DirectionalConv
from src.train import set_seed, get_device, train_model, evaluate

def verify_directional_conv_equivalence():
    conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
    directional = DirectionalConv(3, 5)

    mapping = [
        ("up_left", 0, 0),
        ("up", 0, 1),
        ("up_right", 0, 2),
        ("left", 1, 0),
        ("self", 1, 1),
        ("right", 1, 2),
        ("down_left", 2, 0),
        ("down", 2, 1),
        ("down_right", 2, 2),
    ]

    with torch.no_grad():
        for name, r, c in mapping:
            directional.dir_linears[name].weight.copy_(conv.weight[:, :, r, c])
        directional.bias.copy_(conv.bias)

    x = torch.randn(4, 3, 28, 28)
    expected = conv(x)
    actual = directional(x)
    actual = actual.view(4, 28, 28, 5).permute(0, 3, 1, 2)

    max_error = (expected - actual).abs().max().item()
    print(f"DirectionalConv == Conv2d, max error: {max_error:.2e}")
    assert max_error < 1e-5


def run_gnn_baseline_experiment(
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 32,
    num_layers: int = 2,
    seed: int = 42,
):
    verify_directional_conv_equivalence()
    set_seed(seed)
    device = get_device()
    print(f"Используемое устройство: {device}")

    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir="./data",
        batch_size=batch_size,
        val_size=10000,
        seed=seed
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model = SimpleGNN(num_classes=10, hidden_dim=hidden_dim, num_layers=num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    save_path = f"./results/best_directional_grid8_seed_{seed}.pt"

    print(f"\nСтарт обучения на {epochs} эпох")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        save_path=save_path,
        verbose=True
    )

    print("Оценка на тесте")
    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    os.makedirs("./results", exist_ok=True)
    df_results = pd.DataFrame({
        "epoch": list(range(1, epochs + 1)),
        "train_loss": history["train_loss"],
        "train_acc": history["train_acc"],
        "val_loss": history["val_loss"],
        "val_acc": history["val_acc"],
    })
    df_results["test_acc"] = test_acc
    df_results["model"] = "SimpleGNN_DirectionalGrid8"

    csv_path = f"./results/directional_grid8_seed_{seed}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Результаты эксперимента сохранены в: {csv_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()
    run_gnn_baseline_experiment(
        epochs=20,
        batch_size=64,
        lr=1e-3,
        hidden_dim=32,
        num_layers=args.num_layers,
        seed=args.seed,
    )