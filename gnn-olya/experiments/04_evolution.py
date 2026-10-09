"""
Эксперимент 4: эволюционная оптимизация матрицы смежности без фиксированной плотности
- Число рёбер является частью поиска
- 40 лучших графов порождают 40 потомков через crossover + bit-flip mutation
- При мутации инвертируется около 5% от текущего числа рёбер
- Fitness = лучшая validation accuracy GenericGNN
"""

import argparse
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evolution_utils import (
    add_edges,
    crossover_graphs,
    mutate_bitflip,
    edge_count,
    existing_edges,
    load_checkpoint,
    log_graph_figure,
    # remove_edges,
    # rewire_edges,
    save_checkpoint,
    start_comet_experiment,
)

import pandas as pd
import torch
import torch.nn as nn

from src.data import get_mnist_dataloaders
from src.graph import get_empty_adjacency, get_grid8_adjacency
from src.models import GenericGNN
from src.train import set_seed, get_device, train_model, evaluate


GRAPH_FITNESS_CACHE = {}
mutation_percent = 0.1

def evaluate_graph(adj, train_loader, val_loader, device, epochs, training_seed):
    graph_key = (
        existing_edges(adj).cpu().numpy().tobytes(),
        training_seed,
        epochs,
        train_loader.batch_size,
    )
    if graph_key in GRAPH_FITNESS_CACHE:
        return GRAPH_FITNESS_CACHE[graph_key]

    set_seed(training_seed)

    if train_loader.generator is not None:
        train_loader.generator.manual_seed(training_seed)

    model = GenericGNN(
        adj=adj,
        num_classes=10,
        hidden_dim=128,
        num_layers=3,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)

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
    best_acc = max(history["val_acc"])

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    GRAPH_FITNESS_CACHE[graph_key] = best_acc
    return best_acc


# def mutation_size_for_generation(generation, total_generations):
#     fraction = generation / total_generations
#     if fraction <= 0.25:
#         return 256
#     if fraction <= 0.50:
#         return 128
#     if fraction <= 0.75:
#         return 64
#     return 32


def run_evolution(
    generations=40,
    elite_size=40,
    fitness_epochs=3,
    final_epochs=20,
    batch_size=256,
    seed=42,
    resume=False,
):
    set_seed(seed)
    device = get_device()
    print(f"Устройство: {device}")

    train_loader, val_loader, test_loader = get_mnist_dataloaders(
        data_dir="./data",
        batch_size=batch_size,
        val_size=10000,
        seed=42,
        num_workers=2,
    )

    grid8_adj = get_grid8_adjacency(28, 28)
    max_edges = 784 * 783 // 2
    # bootstrap_edges = [0, 100, 300, 1000, 3000, 10000, 30000, 100000, max_edges]
    bootstrap_edges = [0, 300, 600, 1000, 2000, 3000, 30000, 100000, max_edges]

    history_path = f"./results/evolution_variable_seed_{seed}.csv"
    generation_stats_path = f"./results/evolution_variable_generation_stats_seed_{seed}.csv"
    checkpoint_path = f"./results/evolution_variable_checkpoint_seed_{seed}.pt"
    best_adj_path = f"./results/evolution_variable_best_adj_seed_{seed}.pt"
    figure_dir = f"./results/figures/evolution_variable_seed_{seed}"

    experiment = start_comet_experiment(
        seed,
        {
            "seed": seed,
            "generations": generations,
            "elite_size": elite_size,
            "fitness_epochs": fitness_epochs,
            "final_epochs": final_epochs,
            "batch_size": batch_size,
            "max_edges": max_edges,
            "bootstrap_edges": bootstrap_edges,
            "resume": resume,
        },
    )

    rows = []
    generation_stats_rows = []
    elites = None
    start_generation = 0
    visual_step = 0
    training_seed = seed

    log_graph_figure(
        grid8_adj,
        "reference_grid8",
        experiment,
        figure_dir,
        visual_step,
    )
    visual_step += 1

    if resume:
        if os.path.exists(generation_stats_path):
            generation_stats_rows = pd.read_csv(generation_stats_path).to_dict("records")

        state = load_checkpoint(checkpoint_path, history_path)
        if state is not None:
            start_generation, elites, rows = state
            print(
                f"RESUME: generation={start_generation}, "
                f"best={elites[0]['fitness'] * 100:.2f}%, "
                f"edges={edge_count(elites[0]['adj'])}"
            )
            log_graph_figure(
                elites[0]["adj"],
                f"resume_generation_{start_generation:02d}",
                experiment,
                figure_dir,
                visual_step,
                elites[0]["fitness"],
            )
            visual_step += 1
        else:
            print("Resume checkpoint не найден. Начинаем с нуля.")

    if elites is None:
        empty_graph = get_empty_adjacency(784)
        empty_fitness = evaluate_graph(
            empty_graph,
            train_loader,
            val_loader,
            device,
            fitness_epochs,
            training_seed,
        )

        print(f"Empty graph | val_acc={empty_fitness * 100:.2f}%")

        rows.append({
            "generation": 0,
            "candidate": 0,
            "operation": "empty",
            "parent_rank": -1,
            "mutation_size": 0,
            "edges": 0,
            "val_acc": empty_fitness,
            "selected": False,
        })

        log_graph_figure(
            empty_graph,
            "initial_empty",
            experiment,
            figure_dir,
            visual_step,
            empty_fitness,
        )
        visual_step += 1

        print("\n=== BOOTSTRAP POPULATION ===")
        bootstrap = []
        candidate = 1
        samples_per_density = 10

        for edge_idx in range(len(bootstrap_edges) - 1):
            for sample_idx in range(samples_per_density):
                rng = random.Random(seed * 1_000_000 + edge_idx * 100 + sample_idx)
                edge_min = bootstrap_edges[edge_idx]
                edge_max = bootstrap_edges[edge_idx + 1]
                if edge_idx < len(bootstrap_edges) - 2:
                    edge_max -= 1
                edge_target = rng.randint(edge_min, edge_max)

                adj = add_edges(
                    get_empty_adjacency(784),
                    edge_target,
                    rng,
                    max_edges,
                )
                fitness = evaluate_graph(
                    adj,
                    train_loader,
                    val_loader,
                    device,
                    fitness_epochs,
                    training_seed,
                )

                bootstrap.append({
                    "adj": adj,
                    "fitness": fitness,
                    "operation": "bootstrap",
                    "mutation_size": edge_target,
                    "parent_rank": -1,
                    "candidate": candidate,
                })

                print(
                    f"Bootstrap [{sample_idx+1}/{samples_per_density}] | "
                    f"edges={edge_count(adj):4d} | "
                    f"val_acc={fitness * 100:.2f}%"
                )
                candidate += 1

        bootstrap.sort(key=lambda item: item["fitness"], reverse=True)
        elites = bootstrap[:elite_size]
        selected_ids = {id(item) for item in elites}

        for item in bootstrap:
            rows.append({
                "generation": 0,
                "candidate": item["candidate"],
                "operation": item["operation"],
                "parent_rank": item["parent_rank"],
                "mutation_size": item["mutation_size"],
                "edges": edge_count(item["adj"]),
                "val_acc": item["fitness"],
                "selected": id(item) in selected_ids,
            })

        best = elites[0]
        print(
            f"BEST BOOTSTRAP: edges={edge_count(best['adj'])}, "
            f"val_acc={best['fitness'] * 100:.2f}%"
        )

        population_edges = torch.tensor([edge_count(item["adj"]) for item in bootstrap], dtype=torch.float32)
        generation_stats_rows.append({
            "generation": 0,
            "population_size": len(bootstrap),
            "min_edges": int(population_edges.min().item()),
            "max_edges": int(population_edges.max().item()),
            "mean_edges": population_edges.mean().item(),
            "median_edges": population_edges.quantile(0.5).item(),
            "std_edges": population_edges.std(unbiased=False).item(),
            "best_edges": edge_count(best["adj"]),
            "best_fitness": best["fitness"],
        })
        print(
            f"EDGE STATS: min={generation_stats_rows[-1]['min_edges']}, "
            f"max={generation_stats_rows[-1]['max_edges']}, "
            f"mean={generation_stats_rows[-1]['mean_edges']:.2f}, "
            f"median={generation_stats_rows[-1]['median_edges']:.2f}, "
            f"std={generation_stats_rows[-1]['std_edges']:.2f}"
        )
        pd.DataFrame(generation_stats_rows).to_csv(generation_stats_path, index=False)

        log_graph_figure(
            best["adj"],
            "bootstrap_best",
            experiment,
            figure_dir,
            visual_step,
            best["fitness"],
        )
        visual_step += 1

        torch.save(best["adj"].cpu(), best_adj_path)
        save_checkpoint(checkpoint_path, history_path, 0, elites, rows)

    print("\n=== VARIABLE-DENSITY EVOLUTION ===")

    for generation in range(start_generation + 1, generations + 1):
        print(f"\n--- Generation {generation}/{generations} ---")

        candidates = []
        gen_rng = random.Random(seed * 100_000 + generation)

        for candidate in range(elite_size):
            p1, p2 = gen_rng.sample(elites, 2)

            child_adj = crossover_graphs(p1["adj"], p2["adj"], gen_rng, max_edges)
            child_adj = mutate_bitflip(
                child_adj,
                percent=mutation_percent,
                rng=gen_rng,
                max_edges=max_edges,
            )

            fitness = evaluate_graph(
                child_adj,
                train_loader,
                val_loader,
                device,
                fitness_epochs,
                training_seed,
            )

            candidates.append({
                "adj": child_adj,
                "fitness": fitness,
                "operation": "cross+mut",
                "mutation_size": mutation_percent,
                "parent_rank": -1,
                "candidate": candidate,
            })

            print(
                f"Generation {generation:02d} | "
                f"child={candidate + 1:02d}/{elite_size} | "
                f"edges={edge_count(child_adj):5d} | "
                f"val_acc={fitness * 100:.2f}%"
            )

        pool = [
            {
                "adj": elite["adj"],
                "fitness": elite["fitness"],
                "operation": "elite",
                "mutation_size": 0,
                "parent_rank": i,
                "candidate": -1 - i,
            }
            for i, elite in enumerate(elites)
        ]
        pool.extend(candidates)
        pool.sort(key=lambda item: item["fitness"], reverse=True)

        elites = pool[:elite_size]
        selected_ids = {id(item) for item in elites}

        for item in pool:
            rows.append({
                "generation": generation,
                "candidate": item["candidate"],
                "operation": item["operation"],
                "parent_rank": item["parent_rank"],
                "mutation_size": item["mutation_size"],
                "edges": edge_count(item["adj"]),
                "val_acc": item["fitness"],
                "selected": id(item) in selected_ids,
            })

        best = elites[0]

        print(
            f"BEST GENERATION {generation:02d}: "
            f"edges={edge_count(best['adj'])}, "
            f"val_acc={best['fitness'] * 100:.2f}%"
        )

        population_edges = torch.tensor([edge_count(item["adj"]) for item in pool], dtype=torch.float32)
        generation_stats_rows.append({
            "generation": generation,
            "population_size": len(pool),
            "min_edges": int(population_edges.min().item()),
            "max_edges": int(population_edges.max().item()),
            "mean_edges": population_edges.mean().item(),
            "median_edges": population_edges.quantile(0.5).item(),
            "std_edges": population_edges.std(unbiased=False).item(),
            "best_edges": edge_count(best["adj"]),
            "best_fitness": best["fitness"],
        })
        print(
            f"EDGE STATS: min={generation_stats_rows[-1]['min_edges']}, "
            f"max={generation_stats_rows[-1]['max_edges']}, "
            f"mean={generation_stats_rows[-1]['mean_edges']:.2f}, "
            f"median={generation_stats_rows[-1]['median_edges']:.2f}, "
            f"std={generation_stats_rows[-1]['std_edges']:.2f}"
        )
        pd.DataFrame(generation_stats_rows).to_csv(generation_stats_path, index=False)

        log_graph_figure(
            best["adj"],
            f"generation_{generation:02d}",
            experiment,
            figure_dir,
            visual_step,
            best["fitness"],
        )
        visual_step += 1

        torch.save(best["adj"].cpu(), best_adj_path)
        save_checkpoint(
            checkpoint_path,
            history_path,
            generation,
            elites,
            rows,
        )

    best_graph = elites[0]["adj"]
    best_search_fitness = elites[0]["fitness"]

    pd.DataFrame(
        existing_edges(best_graph).cpu().numpy(),
        columns=["u", "v"],
    ).to_csv(
        f"./results/evolution_variable_best_edges_seed_{seed}.csv",
        index=False,
    )

    print("\n=== FINAL TRAINING OF BEST GRAPH ===")
    set_seed(seed)

    if train_loader.generator is not None:
        train_loader.generator.manual_seed(seed)

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

    model_path = f"./results/best_evolution_variable_model_seed_{seed}.pt"

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

    final_model.load_state_dict(torch.load(model_path, map_location=device))
    _, test_acc = evaluate(final_model, test_loader, criterion, device)

    summary = pd.DataFrame([{
        "seed": seed,
        "generations": generations,
        "elite_size": elite_size,
        "fitness_epochs": fitness_epochs,
        "final_epochs": final_epochs,
        "final_edges": edge_count(best_graph),
        "best_search_val_acc": best_search_fitness,
        "final_best_val_acc": max(final_history["val_acc"]),
        "test_acc": test_acc,
    }])

    summary_path = f"./results/evolution_variable_summary_seed_{seed}.csv"
    summary.to_csv(summary_path, index=False)

    log_graph_figure(
        best_graph,
        "final_best_graph",
        experiment,
        figure_dir,
        visual_step,
        max(final_history["val_acc"]),
    )

    if experiment is not None:
        try:
            experiment.log_metric("final_test_acc", test_acc, step=visual_step)
            experiment.log_metric(
                "final_best_val_acc",
                max(final_history["val_acc"]),
                step=visual_step,
            )
            experiment.log_metric(
                "final_edge_count",
                edge_count(best_graph),
                step=visual_step,
            )
            experiment.end()
        except Exception as e:
            print(f"COMET WARNING: финальное логирование не удалось: {e}")

    print("\nИтог:")
    print(summary.to_string(index=False))
    print(f"История поиска: {history_path}")
    print(f"Лучший граф: {best_adj_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--elite-size", type=int, default=40)
    parser.add_argument("--fitness-epochs", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_evolution(
        generations=args.generations,
        elite_size=args.elite_size,
        fitness_epochs=args.fitness_epochs,
        final_epochs=args.final_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        resume=args.resume,
    )