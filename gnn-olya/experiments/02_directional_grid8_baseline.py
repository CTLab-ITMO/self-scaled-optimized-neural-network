"""
Эксперимент 2: Sanity-бейзлайн Directional Grid-8 GNN на MNIST
- 10 эпох
- Acc = 98.83%
- Результаты в results/directional_grid8.csv
"""

import os
import sys
import pandas as pd
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_mnist_dataloaders
from src.models import SimpleGNN
from src.train import set_seed, get_device, train_model, evaluate


def run_gnn_baseline_experiment(
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    hidden_dim: int = 32,
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

    model = SimpleGNN(num_classes=10, hidden_dim=hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_model_path = "./results/best_directional_grid8.pt"

    print(f"\nСтарт обучения на {epochs} эпох")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        save_path=best_model_path,
        verbose=True
    )

    print("Оценка на тесте")
    model.load_state_dict(torch.load(best_model_path))
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

    csv_path = "./results/directional_grid8.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Результаты эксперимента сохранены в: {csv_path}\n")


if __name__ == "__main__":
    run_gnn_baseline_experiment(epochs=10, batch_size=64, lr=1e-3, hidden_dim=32, seed=42)