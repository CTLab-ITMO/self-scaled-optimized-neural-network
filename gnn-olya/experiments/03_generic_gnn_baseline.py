"""
Эксперимент 3: Проверка чувствительности GenericGNN к структуре графа A
Сравниваем 3 варианта: Empty graph (A = 0), Random graph, Grid-8 graph
"""

import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_mnist_dataloaders
from src.graph import get_empty_adjacency, get_random_adjacency, get_grid8_adjacency
from src.models import GenericGNN
from src.train import set_seed, get_device, train_model, evaluate


def run_generic_gnn_comparison(epochs: int = 20, batch_size: int = 64, lr: float = 1e-3, seed: int = 42):
    set_seed(seed)
    device = get_device()
    print(f"Устройство: {device}")

    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir="./data", batch_size=batch_size, val_size=10000, seed=seed
    )

    grid8_adj = get_grid8_adjacency(28, 28)
    num_grid_edges = int(grid8_adj.sum().item() // 2)

    graphs = {
        "empty": get_empty_adjacency(784),
        "random": get_random_adjacency(784, num_edges=num_grid_edges, seed=seed),
        "grid8": grid8_adj,
    }

    results = []

    for graph_name, adj in graphs.items():
        print(f"\nОбучение GenericGNN на графе: [{graph_name}] (рёбер: {int(adj.sum().item() // 2)})")
        set_seed(seed)

        model = GenericGNN(adj=adj, num_classes=10, hidden_dim=128, num_layers=3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        save_path = f"./results/best_generic_{graph_name}_seed_{seed}.pt"
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

        model.load_state_dict(torch.load(save_path))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"[{graph_name.upper()}] Test Accuracy: {test_acc * 100:.2f}%")

        results.append({
            "graph_type": graph_name,
            "edges": int(adj.sum().item() // 2),
            "best_val_acc": max(history["val_acc"]),
            "test_acc": test_acc
        })

    os.makedirs("./results", exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = f"./results/generic_gnn_seed_{seed}.csv"
    df.to_csv(csv_path, index=False)

    print("\nИтоговая сводка:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_generic_gnn_comparison(epochs=20, batch_size=64, lr=1e-3, seed=args.seed)