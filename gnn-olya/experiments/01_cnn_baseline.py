"""
Эксперимент 1: SimpleCNN на MNIST
- 20 эпох
- Acc = 99.23%
- Результаты в results/cnn_baseline.csv
"""

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_mnist_dataloaders
from src.models import SimpleCNN
from src.train import set_seed, get_device, train_model, evaluate


def run_cnn_experiment(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42
):
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

    model = SimpleCNN(num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    save_path = f"./results/best_cnn_seed_{seed}.pt"

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
    model.load_state_dict(torch.load(save_path))
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
    df_results["model"] = "SimpleCNN"

    csv_path = f"./results/cnn_baseline_seed_{seed}.csv"
    df_results.to_csv(csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_cnn_experiment(epochs=20, batch_size=64, lr=1e-3, seed=args.seed)