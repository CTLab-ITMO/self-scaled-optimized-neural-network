"""
Строит диагностические графики для evolutionary search.

Читает:
- results/evolution_variable_seed_*.csv
- results/evolution_variable_generation_stats_seed_*.csv

Сохраняет для каждого seed в:
- results/figures/evolution_variable_seed_<seed>/diagnostic_01_fitness_progress.png
- results/figures/evolution_variable_seed_<seed>/diagnostic_02_edge_count_progress.png
- results/figures/evolution_variable_seed_<seed>/diagnostic_03_fitness_vs_edges.png
- results/figures/evolution_variable_seed_<seed>/diagnostic_04_children_selection.png
- results/figures/evolution_variable_seed_<seed>/diagnostic_05_new_candidate_fitness_distribution.png
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")


def get_seed(path: Path) -> int:
    match = re.search(r"_seed_(\d+)\.csv$", path.name)
    if match is None:
        raise ValueError(f"Не удалось извлечь seed из {path}")
    return int(match.group(1))


def load_history(seed: int) -> pd.DataFrame:
    path = RESULTS_DIR / f"evolution_variable_seed_{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")

    df = pd.read_csv(path)
    required = {"generation", "operation", "edges", "val_acc", "selected"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"В {path} не хватает колонок: {sorted(missing)}")

    if df["selected"].dtype != bool:
        df["selected"] = df["selected"].astype(str).str.lower().eq("true")

    return df


def load_generation_stats(seed: int, history: pd.DataFrame) -> pd.DataFrame:
    path = RESULTS_DIR / f"evolution_variable_generation_stats_seed_{seed}.csv"
    if path.exists():
        return pd.read_csv(path)

    rows = []
    for generation, group in history.groupby("generation"):
        population = group[group["operation"] != "empty"]
        if population.empty:
            continue

        edges = population["edges"].to_numpy(dtype=float)
        best = population.loc[population["val_acc"].idxmax()]
        rows.append({
            "generation": int(generation),
            "population_size": len(population),
            "min_edges": float(np.min(edges)),
            "max_edges": float(np.max(edges)),
            "mean_edges": float(np.mean(edges)),
            "median_edges": float(np.median(edges)),
            "std_edges": float(np.std(edges)),
            "best_edges": float(best["edges"]),
            "best_fitness": float(best["val_acc"]),
        })

    return pd.DataFrame(rows).sort_values("generation")


def plot_fitness_progress(history: pd.DataFrame, stats: pd.DataFrame, figure_dir: Path):
    population = history[history["operation"] != "empty"].copy()
    pool_stats = (
        population.groupby("generation")["val_acc"]
        .agg(["mean", "median"])
        .reset_index()
    )

    new_candidates = population[population["operation"].isin(["bootstrap", "cross+mut"])]
    best_new = (
        new_candidates.groupby("generation")["val_acc"]
        .max()
        .rename("best_new")
        .reset_index()
    )

    data = stats[["generation", "best_fitness"]].merge(pool_stats, on="generation", how="left")
    data = data.merge(best_new, on="generation", how="left")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(data["generation"], data["best_fitness"] * 100.0, marker="o", label="Best overall")
    ax.plot(data["generation"], data["best_new"] * 100.0, marker="o", label="Best new candidate")
    ax.plot(data["generation"], data["mean"] * 100.0, label="Population mean")
    ax.plot(data["generation"], data["median"] * 100.0, label="Population median")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation accuracy, %")
    ax.set_title("Evolution fitness by generation")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = figure_dir / "diagnostic_01_fitness_progress.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_edge_count_progress(stats: pd.DataFrame, figure_dir: Path):
    data = stats.sort_values("generation")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.fill_between(
        data["generation"],
        data["min_edges"],
        data["max_edges"],
        alpha=0.15,
        label="Min-max",
    )
    ax.plot(data["generation"], data["mean_edges"], marker="o", label="Mean")
    ax.plot(data["generation"], data["median_edges"], marker="o", label="Median")
    ax.plot(data["generation"], data["best_edges"], marker="o", label="Best individual")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of edges")
    ax.set_title("Edge-count evolution")
    ax.set_yscale("symlog", linthresh=10)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = figure_dir / "diagnostic_02_edge_count_progress.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_fitness_vs_edges(history: pd.DataFrame, figure_dir: Path):
    data = history[
        history["operation"].isin(["bootstrap", "cross+mut"])
        & (history["edges"] > 0)
    ].copy()

    selected = data[data["selected"]]
    rejected = data[~data["selected"]]

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.scatter(
        rejected["edges"],
        rejected["val_acc"] * 100.0,
        s=20,
        alpha=0.35,
        label="Rejected",
    )
    ax.scatter(
        selected["edges"],
        selected["val_acc"] * 100.0,
        s=28,
        alpha=0.75,
        label="Selected",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Number of edges, log scale")
    ax.set_ylabel("Validation accuracy, %")
    ax.set_title("Fitness vs graph density")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = figure_dir / "diagnostic_03_fitness_vs_edges.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_children_selection(history: pd.DataFrame, stats: pd.DataFrame, figure_dir: Path):
    children = history[history["operation"] == "cross+mut"].copy()
    generations = sorted(children["generation"].unique())

    selected_counts = []
    better_than_previous_best = []

    previous_best = stats.set_index("generation")["best_fitness"].to_dict()

    for generation in generations:
        group = children[children["generation"] == generation]
        selected_counts.append(int(group["selected"].sum()))

        prev = previous_best.get(generation - 1)
        if prev is None:
            better_than_previous_best.append(0)
        else:
            better_than_previous_best.append(int((group["val_acc"] > prev).sum()))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(generations, selected_counts, marker="o", label="Children selected into top-40")
    ax.plot(generations, better_than_previous_best, marker="o", label="Children beating previous best")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Number of children")
    ax.set_title("How much useful variation each generation produces")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = figure_dir / "diagnostic_04_children_selection.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_new_candidate_fitness_distribution(history: pd.DataFrame, figure_dir: Path):
    data = history[history["operation"].isin(["bootstrap", "cross+mut"])].copy()
    generations = sorted(data["generation"].unique())
    values = [
        data.loc[data["generation"] == generation, "val_acc"].to_numpy() * 100.0
        for generation in generations
    ]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.boxplot(values, showfliers=False)
    ax.set_xticks(np.arange(1, len(generations) + 1))
    ax.set_xticklabels([str(generation) for generation in generations])

    ax.set_xlabel("Generation")
    ax.set_ylabel("Validation accuracy, %")
    ax.set_title("Fitness distribution of newly evaluated candidates")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = figure_dir / "diagnostic_05_new_candidate_fitness_distribution.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def visualize_seed(seed: int):
    history = load_history(seed)
    stats = load_generation_stats(seed, history)

    figure_dir = RESULTS_DIR / "figures" / f"evolution_variable_seed_{seed}"
    figure_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_fitness_progress(history, stats, figure_dir),
        plot_edge_count_progress(stats, figure_dir),
        plot_fitness_vs_edges(history, figure_dir),
        plot_children_selection(history, stats, figure_dir),
        plot_new_candidate_fitness_distribution(history, figure_dir),
    ]

    print(f"\nseed={seed}")
    for path in paths:
        print(f"saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = sorted({
            get_seed(path)
            for path in RESULTS_DIR.glob("evolution_variable_seed_*.csv")
        })

    if not seeds:
        raise FileNotFoundError("Не найдены results/evolution_variable_seed_*.csv")

    for seed in seeds:
        visualize_seed(seed)


if __name__ == "__main__":
    main()
