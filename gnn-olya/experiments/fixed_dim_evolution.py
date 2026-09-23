"""
Эксперимент 4: эволюционная оптимизация матрицы смежности.

Стартуем с пустого графа и постепенно увеличиваем число рёбер до плотности Grid-8.
После этого фиксируем число рёбер и оптимизируем только топологию через rewire.
Fitness каждого графа = лучшая validation accuracy GenericGNN.
Test set используется только один раз — для финального лучшего графа.
"""

import argparse
import os
import random
import sys

import pandas as pd
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import get_mnist_dataloaders
from src.graph import get_empty_adjacency, get_grid8_adjacency
from src.models import GenericGNN
from src.train import set_seed, get_device, train_model, evaluate


def edge_count(adj: torch.Tensor) -> int:
    return int(adj.sum().item() // 2)


def existing_edges(adj: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(torch.triu(adj, diagonal=1), as_tuple=False)


def start_comet_experiment(seed: int, parameters: dict):
    api_key = os.environ.get("COMET_API_KEY")

    if not api_key:
        print("COMET WARNING: COMET_API_KEY не задан")
        return None

    try:
        from comet_ml import start

        experiment = start(
            api_key=api_key,
            workspace=os.environ.get("COMET_WORKSPACE"),
            project_name=os.environ.get(
                "COMET_PROJECT_NAME",
                "gnn-experiments"
            ),
        )

        experiment.set_name(f"evolution_seed_{seed}")
        experiment.log_parameters(parameters)

        return experiment

    except Exception as e:
        print(f"COMET WARNING: не удалось запустить Comet: {e}")
        return None


def make_graph_figure(
    adj: torch.Tensor,
    title: str,
    val_acc: float | None = None,
    height: int = 28,
    width: int = 28,
):
    edges = existing_edges(adj).cpu().numpy()

    node_ids = np.arange(height * width)
    x = node_ids % width
    y = (height - 1) - (node_ids // width)

    fig, ax = plt.subplots(figsize=(7, 7))

    if len(edges) > 0:
        u = edges[:, 0]
        v = edges[:, 1]

        segments = np.stack(
            [
                np.column_stack([u % width, (height - 1) - (u // width)]),
                np.column_stack([v % width, (height - 1) - (v // width)]),
            ],
            axis=1,
        )

        ax.add_collection(
            LineCollection(
                segments,
                linewidths=0.35,
                alpha=0.16,
            )
        )

    ax.scatter(x, y, s=7, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-1, width)
    ax.set_ylim(-1, height)
    ax.axis("off")

    subtitle = f"edges={edge_count(adj)}"
    if val_acc is not None:
        subtitle += f", val_acc={val_acc * 100:.2f}%"

    ax.set_title(f"{title}\n{subtitle}")
    fig.tight_layout()
    return fig


def log_graph_figure(
    adj: torch.Tensor,
    name: str,
    experiment,
    figure_dir: str,
    step: int,
    val_acc: float | None = None,
):
    os.makedirs(figure_dir, exist_ok=True)

    fig = make_graph_figure(
        adj=adj,
        title=name,
        val_acc=val_acc,
    )

    path = os.path.join(figure_dir, f"{name}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if experiment is not None:
        try:
            experiment.log_image(
                image_data=path,
                name=f"{name}.png",
                step=step,
            )

            if val_acc is not None:
                experiment.log_metric(
                    "best_val_acc",
                    val_acc,
                    step=step,
                )

            experiment.log_metric(
                "edge_count",
                edge_count(adj),
                step=step,
            )

        except Exception as e:
            print(f"COMET WARNING: не удалось залогировать {name}: {e}")

    return path


def add_edges_to_target(adj: torch.Tensor, target_edges: int, rng: random.Random) -> torch.Tensor:
    out = adj.clone()
    n = out.size(0)
    need = target_edges - edge_count(out)

    if need <= 0:
        return out

    added = 0
    while added < need:
        u = rng.randrange(n)
        v = rng.randrange(n - 1)
        if v >= u:
            v += 1

        if out[u, v] == 0:
            out[u, v] = 1.0
            out[v, u] = 1.0
            added += 1

    return out


def rewire_edges(adj: torch.Tensor, count: int, rng: random.Random) -> torch.Tensor:
    out = adj.clone()
    edges = existing_edges(out)

    count = min(count, len(edges))
    if count == 0:
        return out

    chosen = rng.sample(range(len(edges)), count)

    for idx in chosen:
        u, v = edges[idx].tolist()
        out[u, v] = 0.0
        out[v, u] = 0.0

    return add_edges_to_target(out, edge_count(adj), rng)


def evaluate_graph(
    adj: torch.Tensor,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int,
    training_seed: int,
) -> float:
    set_seed(training_seed)

    model = GenericGNN(
        adj=adj,
        num_classes=10,
        hidden_dim=128,
        num_layers=3,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        save_path=None,
        verbose=False,
    )

    return max(history["val_acc"])


def mutation_size_for_generation(generation: int, total_generations: int) -> int:
    fraction = generation / total_generations

    if fraction <= 0.25:
        return 256
    if fraction <= 0.50:
        return 128
    if fraction <= 0.75:
        return 64
    return 32


def save_history(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)



def save_checkpoint(
    path: str,
    phase: str,
    growth_step: int,
    topology_generation: int,
    elites: list[dict],
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "phase": phase,
        "growth_step": growth_step,
        "topology_generation": topology_generation,
        "elites": [
            {"adj": elite["adj"].cpu(), "fitness": float(elite["fitness"])}
            for elite in elites
        ],
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_history(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return pd.read_csv(path).to_dict("records")


def load_resume_state(checkpoint_path: str, history_path: str, best_adj_path: str):
    rows = load_history(history_path)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        elites = [
            {"adj": item["adj"].float(), "fitness": float(item["fitness"])}
            for item in checkpoint["elites"]
        ]
        return (
            checkpoint["phase"],
            int(checkpoint["growth_step"]),
            int(checkpoint["topology_generation"]),
            elites,
            rows,
        )

    if os.path.exists(history_path) and os.path.exists(best_adj_path) and rows:
        df = pd.DataFrame(rows)
        best_adj = torch.load(best_adj_path, map_location="cpu").float()
        topology_rows = df[df["phase"] == "topology"]
        growth_rows = df[df["phase"] == "growth"]

        if not topology_rows.empty:
            last_topology = int(topology_rows["step"].max())
            step_rows = topology_rows[topology_rows["step"] == last_topology]
            best_fitness = float(step_rows["val_acc"].max())
            last_growth = int(growth_rows["step"].max()) if not growth_rows.empty else 0
            print(
                "Найден результат предыдущей версии без top-3 checkpoint. "
                f"Продолжаем с лучшего графа после topology generation {last_topology}."
            )
            return (
                "topology",
                last_growth,
                last_topology,
                [{"adj": best_adj, "fitness": best_fitness}],
                rows,
            )

        if not growth_rows.empty:
            last_growth = int(growth_rows["step"].max())
            step_rows = growth_rows[growth_rows["step"] == last_growth]
            best_fitness = float(step_rows["val_acc"].max())
            print(
                "Найден результат предыдущей версии без top-3 checkpoint. "
                f"Продолжаем с лучшего графа после growth step {last_growth}."
            )
            return (
                "growth",
                last_growth,
                0,
                [{"adj": best_adj, "fitness": best_fitness}],
                rows,
            )

    return None

def run_evolution(
    growth_targets: list[int] | None = None,
    growth_candidates: int = 6,
    topology_generations: int = 15,
    elite_size: int = 3,
    children_per_elite: int = 3,
    fitness_epochs: int = 20,
    final_epochs: int = 20,
    batch_size: int = 64,
    seed: int = 42,
    resume: bool = False,
):
    set_seed(seed)
    device = get_device()
    print(f"Устройство: {device}")

    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir="./data",
        batch_size=batch_size,
        val_size=10000,
        seed=seed,
    )

    grid8_adj = get_grid8_adjacency(28, 28)
    max_edges = edge_count(grid8_adj)

    if growth_targets is None:
        growth_targets = [500, 1000, 1500, 2000, 2500, max_edges]

    growth_targets = sorted({
        min(target, max_edges)
        for target in growth_targets
        if target > 0
    })

    if not growth_targets or growth_targets[-1] != max_edges:
        growth_targets.append(max_edges)

    history_path = f"./results/evolution_seed_{seed}.csv"
    best_adj_path = f"./results/evolution_best_adj_seed_{seed}.pt"
    checkpoint_path = f"./results/evolution_checkpoint_seed_{seed}.pt"
    figure_dir = f"./results/figures/evolution_seed_{seed}"

    experiment = start_comet_experiment(
        seed=seed,
        parameters={
            "seed": seed,
            "fitness_epochs": fitness_epochs,
            "final_epochs": final_epochs,
            "batch_size": batch_size,
            "growth_candidates": growth_candidates,
            "topology_generations": topology_generations,
            "elite_size": elite_size,
            "children_per_elite": children_per_elite,
            "max_edges": max_edges,
            "resume": resume,
        },
    )

    visual_step = 0

    log_graph_figure(
        adj=grid8_adj,
        name="reference_grid8",
        experiment=experiment,
        figure_dir=figure_dir,
        step=visual_step,
    )
    visual_step += 1

    rows = []
    training_seed = seed
    start_growth_step = 0
    start_topology_generation = 0
    elites = None

    if resume:
        state = load_resume_state(
            checkpoint_path=checkpoint_path,
            history_path=history_path,
            best_adj_path=best_adj_path,
        )
        if state is None:
            print("Resume-файлы не найдены. Начинаем с нуля.")
        else:
            (
                phase,
                start_growth_step,
                start_topology_generation,
                elites,
                rows,
            ) = state
            print(
                f"RESUME: phase={phase}, growth_step={start_growth_step}, "
                f"topology_generation={start_topology_generation}, elites={len(elites)}"
            )

            log_graph_figure(
                adj=elites[0]["adj"],
                name=f"resume_{phase}_g{start_growth_step}_t{start_topology_generation}",
                experiment=experiment,
                figure_dir=figure_dir,
                step=visual_step,
                val_acc=elites[0]["fitness"],
            )
            visual_step += 1

    print(
        f"Старт: {'resume' if elites is not None else 'empty'} | "
        f"fitness epochs: {fitness_epochs} | max edges: {max_edges}"
    )
    print(
        f"Growth targets: {growth_targets} | growth candidates: {growth_candidates}"
    )
    print(
        f"Topology generations: {topology_generations} | elite: {elite_size} | "
        f"children per elite: {children_per_elite}"
    )

    if elites is None:
        initial_graph = get_empty_adjacency(784)
        initial_fitness = evaluate_graph(
            initial_graph,
            train_loader,
            val_loader,
            device,
            fitness_epochs,
            training_seed,
        )

        print(
            f"\nInitial empty graph | edges=0 | val_acc={initial_fitness * 100:.2f}%"
        )

        log_graph_figure(
            adj=initial_graph,
            name="initial_empty",
            experiment=experiment,
            figure_dir=figure_dir,
            step=visual_step,
            val_acc=initial_fitness,
        )
        visual_step += 1

        rows.append({
            "phase": "initial",
            "step": 0,
            "candidate": 0,
            "parent_rank": 0,
            "mutation_size": 0,
            "edges": 0,
            "val_acc": initial_fitness,
            "selected": True,
        })

        elites = [{
            "adj": initial_graph,
            "fitness": initial_fitness,
        }]

        save_history(rows, history_path)
        save_checkpoint(
            checkpoint_path,
            phase="initial",
            growth_step=0,
            topology_generation=0,
            elites=elites,
        )

    # Phase 1: growth.
    for growth_step, target_edges in enumerate(growth_targets, start=1):
        if growth_step <= start_growth_step:
            continue

        print(
            f"\n=== GROWTH {growth_step}/{len(growth_targets)}: "
            f"target={target_edges} edges ==="
        )

        candidates = []

        for candidate_idx in range(growth_candidates):
            parent_rank = candidate_idx % len(elites)
            parent = elites[parent_rank]

            rng = random.Random(
                seed * 1_000_000
                + growth_step * 10_000
                + candidate_idx
            )

            child = add_edges_to_target(parent["adj"], target_edges, rng)

            fitness = evaluate_graph(
                child,
                train_loader,
                val_loader,
                device,
                fitness_epochs,
                training_seed,
            )

            candidates.append({
                "adj": child,
                "fitness": fitness,
                "parent_rank": parent_rank,
                "candidate": candidate_idx,
            })

            print(
                f"Growth {growth_step:02d} | "
                f"candidate={candidate_idx:02d} | "
                f"parent={parent_rank} | "
                f"edges={edge_count(child):4d} | "
                f"val_acc={fitness * 100:.2f}%"
            )

        ranked = sorted(
            candidates,
            key=lambda item: item["fitness"],
            reverse=True,
        )

        selected = ranked[:min(elite_size, len(ranked))]
        selected_ids = {item["candidate"] for item in selected}

        for item in candidates:
            parent_edges = edge_count(elites[item["parent_rank"]]["adj"])
            rows.append({
                "phase": "growth",
                "step": growth_step,
                "candidate": item["candidate"],
                "parent_rank": item["parent_rank"],
                "mutation_size": target_edges - parent_edges,
                "edges": edge_count(item["adj"]),
                "val_acc": item["fitness"],
                "selected": item["candidate"] in selected_ids,
            })

        elites = [
            {
                "adj": item["adj"],
                "fitness": item["fitness"],
            }
            for item in selected
        ]

        best = elites[0]

        print(
            f"BEST GROWTH {growth_step:02d}: "
            f"edges={edge_count(best['adj'])}, "
            f"val_acc={best['fitness'] * 100:.2f}%"
        )

        log_graph_figure(
            adj=best["adj"],
            name=f"growth_{growth_step:02d}_{edge_count(best['adj'])}_edges",
            experiment=experiment,
            figure_dir=figure_dir,
            step=visual_step,
            val_acc=best["fitness"],
        )
        visual_step += 1

        save_history(rows, history_path)
        torch.save(best["adj"].cpu(), best_adj_path)
        save_checkpoint(
            checkpoint_path,
            phase="growth",
            growth_step=growth_step,
            topology_generation=0,
            elites=elites,
        )
        start_growth_step = growth_step

    # Phase 2: fixed-density topology search.
    print(
        f"\n=== FIXED-DENSITY TOPOLOGY SEARCH: {max_edges} edges ==="
    )

    for generation in range(start_topology_generation + 1, topology_generations + 1):
        mutation_size = mutation_size_for_generation(
            generation,
            topology_generations,
        )

        print(
            f"\n--- Topology generation {generation}/{topology_generations} | "
            f"rewire={mutation_size} ---"
        )

        children = []
        child_id = 0

        for parent_rank, parent in enumerate(elites):
            for local_child in range(children_per_elite):
                rng = random.Random(
                    seed * 10_000_000
                    + generation * 100_000
                    + parent_rank * 1_000
                    + local_child
                )

                child = rewire_edges(
                    parent["adj"],
                    mutation_size,
                    rng,
                )

                fitness = evaluate_graph(
                    child,
                    train_loader,
                    val_loader,
                    device,
                    fitness_epochs,
                    training_seed,
                )

                children.append({
                    "adj": child,
                    "fitness": fitness,
                    "parent_rank": parent_rank,
                    "candidate": child_id,
                })

                print(
                    f"Topology {generation:02d} | "
                    f"candidate={child_id:02d} | "
                    f"parent={parent_rank} | "
                    f"rewire={mutation_size:3d} | "
                    f"val_acc={fitness * 100:.2f}%"
                )

                child_id += 1

        pool = []

        for parent_rank, elite in enumerate(elites):
            pool.append({
                "adj": elite["adj"],
                "fitness": elite["fitness"],
                "source": "elite",
                "parent_rank": parent_rank,
                "candidate": -1 - parent_rank,
            })

        for child in children:
            pool.append({
                **child,
                "source": "child",
            })

        ranked = sorted(
            pool,
            key=lambda item: item["fitness"],
            reverse=True,
        )

        selected = ranked[:elite_size]
        selected_keys = {
            (item["source"], item["candidate"])
            for item in selected
        }

        for item in pool:
            rows.append({
                "phase": "topology",
                "step": generation,
                "candidate": item["candidate"],
                "parent_rank": item["parent_rank"],
                "mutation_size": 0 if item["source"] == "elite" else mutation_size,
                "edges": edge_count(item["adj"]),
                "val_acc": item["fitness"],
                "selected": (item["source"], item["candidate"]) in selected_keys,
            })

        elites = [
            {
                "adj": item["adj"],
                "fitness": item["fitness"],
            }
            for item in selected
        ]

        best = elites[0]

        print(
            f"BEST TOPOLOGY {generation:02d}: "
            f"edges={edge_count(best['adj'])}, "
            f"val_acc={best['fitness'] * 100:.2f}%"
        )

        log_graph_figure(
            adj=best["adj"],
            name=f"topology_{generation:02d}",
            experiment=experiment,
            figure_dir=figure_dir,
            step=visual_step,
            val_acc=best["fitness"],
        )
        visual_step += 1

        save_history(rows, history_path)
        torch.save(best["adj"].cpu(), best_adj_path)
        save_checkpoint(
            checkpoint_path,
            phase="topology",
            growth_step=len(growth_targets),
            topology_generation=generation,
            elites=elites,
        )

    best_graph = elites[0]["adj"]
    best_search_fitness = elites[0]["fitness"]

    best_edges = existing_edges(best_graph)
    pd.DataFrame(
        best_edges.cpu().numpy(),
        columns=["u", "v"],
    ).to_csv(
        f"./results/evolution_best_edges_seed_{seed}.csv",
        index=False,
    )

    # Final evaluation.
    print("\n=== FINAL TRAINING OF BEST GRAPH ===")

    set_seed(seed)

    final_model = GenericGNN(
        adj=best_graph,
        num_classes=10,
        hidden_dim=128,
        num_layers=3,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        final_model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    model_path = f"./results/best_evolution_model_seed_{seed}.pt"

    final_history = train_model(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=final_epochs,
        device=device,
        save_path=model_path,
        verbose=True,
    )

    final_model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    test_loss, test_acc = evaluate(
        final_model,
        test_loader,
        criterion,
        device,
    )

    summary = pd.DataFrame([{
        "seed": seed,
        "fitness_epochs": fitness_epochs,
        "final_epochs": final_epochs,
        "growth_candidates": growth_candidates,
        "topology_generations": topology_generations,
        "elite_size": elite_size,
        "children_per_elite": children_per_elite,
        "final_edges": edge_count(best_graph),
        "best_search_val_acc": best_search_fitness,
        "final_best_val_acc": max(final_history["val_acc"]),
        "test_acc": test_acc,
    }])

    summary_path = f"./results/evolution_summary_seed_{seed}.csv"
    summary.to_csv(summary_path, index=False)

    log_graph_figure(
        adj=best_graph,
        name="final_best_graph",
        experiment=experiment,
        figure_dir=figure_dir,
        step=visual_step,
        val_acc=max(final_history["val_acc"]),
    )

    if experiment is not None:
        try:
            experiment.log_metric(
                "final_test_acc",
                test_acc,
                step=visual_step,
            )
            experiment.log_metric(
                "final_best_val_acc",
                max(final_history["val_acc"]),
                step=visual_step,
            )
            experiment.end()

        except Exception as e:
            print(f"COMET WARNING: финальное логирование не удалось: {e}")

    print("\nИтог:")
    print(summary.to_string(index=False))
    print(f"\nИстория поиска: {history_path}")
    print(f"Лучший граф: {best_adj_path}")
    print(
        f"Рёбра лучшего графа: "
        f"./results/evolution_best_edges_seed_{seed}.csv"
    )
    print(f"Итоговая модель: {model_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fitness-epochs", type=int, default=20)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--growth-candidates", type=int, default=6)
    parser.add_argument("--topology-generations", type=int, default=15)
    parser.add_argument("--elite-size", type=int, default=3)
    parser.add_argument("--children-per-elite", type=int, default=3)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    run_evolution(
        growth_candidates=args.growth_candidates,
        topology_generations=args.topology_generations,
        elite_size=args.elite_size,
        children_per_elite=args.children_per_elite,
        fitness_epochs=args.fitness_epochs,
        final_epochs=args.final_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        resume=args.resume,
    )
