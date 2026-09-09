# рисует графики, сохраняет локально и отправляет в Comet

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

MAIN_GRAPHS = [
    "knn_corr_k8",
    "grid4",
    "er_grid4",
    "empty",
    "grid4_misaligned",
    "er_knn8",
    "complete",
]

METHOD_LABELS = {
    "corr_static": "Correlation",
    "partial_static": "Partial correlation (fixed size)",
    "partial_free": "Partial correlation (chooses size)",
}

RULE_LABELS = {
    "keep": "Keep graph",
    "random_rewire": "Random rewire",
    "corr_static": "Correlation",
    "partial_static": "Partial correlation (fixed size)",
    "partial_free": "Partial correlation (chooses size)",
}

GRAPH_LABELS = {
    "knn_corr_k8": "kNN correlation",
    "grid4": "Grid",
    "er_grid4": "Random, same density as grid",
    "empty": "Empty",
    "grid4_misaligned": "Grid, wrong alignment",
    "er_knn8": "Random, same density as kNN",
    "complete": "Complete",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all result visualizations and log them to Comet ML."
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("COMET_PROJECT_NAME", "gnn-results"),
        help="Comet project name (default: COMET_PROJECT_NAME or gnn-results)",
    )
    parser.add_argument(
        "--name",
        default=os.environ.get("COMET_EXPERIMENT_NAME", "results-visualization"),
        help="Comet experiment name",
    )
    parser.add_argument(
        "--no-comet",
        action="store_true",
        help="Only save PNG files locally; do not connect to Comet ML",
    )
    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {
        "main": RESULTS_DIR / "results.csv",
        "structure": RESULTS_DIR / "structure.csv",
        "refine": RESULTS_DIR / "refine.csv",
    }
    for path in paths.values():
        require_file(path)
    return (
        pd.read_csv(paths["main"]),
        pd.read_csv(paths["structure"]),
        pd.read_csv(paths["refine"]),
    )


def start_comet(args: argparse.Namespace):
    if args.no_comet:
        return None

    api_key = os.environ.get("COMET_API_KEY")
    workspace = os.environ.get("COMET_WORKSPACE")
    if not api_key:
        raise RuntimeError(
            "COMET_API_KEY is not set. Set it in the shell or run with --no-comet."
        )
    if not workspace:
        raise RuntimeError(
            "COMET_WORKSPACE is not set. Set it in the shell or run with --no-comet."
        )

    from comet_ml import start

    experiment = start(
        api_key=api_key,
        workspace=workspace,
        project_name=args.project,
    )
    experiment.set_name(args.name)
    experiment.log_parameters(
        {
            "visualization_source": "existing CSV results",
            "main_results": "results/results.csv",
            "structure_results": "results/structure.csv",
            "refine_results": "results/refine.csv",
        }
    )
    return experiment


def save_and_log(fig: plt.Figure, filename: str, experiment) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight")
    if experiment is not None:
        experiment.log_figure(figure_name=path.stem, figure=fig)
    plt.close(fig)
    print(f"saved: {path.relative_to(ROOT)}")
    return path


def plot_main_graph_accuracy(df: pd.DataFrame, experiment) -> None:
    data = df[df["graph"].isin(MAIN_GRAPHS)].copy()
    missing = sorted(set(MAIN_GRAPHS) - set(data["graph"].unique()))
    if missing:
        raise ValueError(f"Missing main graphs in results.csv: {missing}")

    summary = (
        data.groupby("graph")["test_accuracy"]
        .agg(["mean", "std"])
        .reindex(MAIN_GRAPHS)
        .sort_values("mean", ascending=True)
    )

    labels = [GRAPH_LABELS.get(g, g) for g in summary.index]
    means = summary["mean"].to_numpy() * 100.0
    stds = summary["std"].fillna(0).to_numpy() * 100.0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(summary))
    ax.barh(y, means, xerr=stds, capsize=4)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Test accuracy, %")
    ax.set_title("Main GNN result: accuracy strongly depends on the graph")
    ax.set_xlim(0, max(60, float(means.max() + 6)))
    ax.grid(axis="x", alpha=0.25)

    for i, value in enumerate(means):
        ax.text(value + 0.7, i, f"{value:.1f}%", va="center")

    fig.tight_layout()
    save_and_log(fig, "01_main_graph_accuracy.png", experiment)


def plot_alignment_comparison(df: pd.DataFrame, experiment) -> float:
    pair = df[df["graph"].isin(["grid4", "grid4_misaligned"])][
        ["graph", "seed", "test_accuracy"]
    ].copy()
    pivot = pair.pivot_table(index="seed", columns="graph", values="test_accuracy", aggfunc="mean")
    pivot = pivot.dropna(subset=["grid4", "grid4_misaligned"])
    if pivot.empty:
        raise ValueError("No paired grid4/grid4_misaligned seeds found")

    aligned = pivot["grid4"].to_numpy() * 100.0
    misaligned = pivot["grid4_misaligned"].to_numpy() * 100.0
    gap = float((aligned - misaligned).mean())

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for a, b in zip(aligned, misaligned):
        ax.plot([0, 1], [a, b], marker="o", alpha=0.55)

    means = [aligned.mean(), misaligned.mean()]
    ax.scatter([0, 1], means, s=130, marker="D", zorder=5, label="Mean")
    ax.set_xticks([0, 1], ["Correct alignment", "Same graph, wrong alignment"])
    ax.set_ylabel("Test accuracy, %")
    ax.set_title(f"Wrong node-to-pixel alignment costs {gap:.1f} percentage points")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_and_log(fig, "02_alignment_comparison.png", experiment)
    return gap


def plot_density_controls(df: pd.DataFrame, experiment) -> dict[str, float]:
    graphs = ["grid4", "er_grid4", "knn_corr_k8", "er_knn8"]
    data = df[df["graph"].isin(graphs)].copy()
    summary = data.groupby("graph")["test_accuracy"].agg(["mean", "std"])
    missing = sorted(set(graphs) - set(summary.index))
    if missing:
        raise ValueError(f"Missing density-control graphs: {missing}")

    pairs = [
        ("Grid", "grid4", "er_grid4"),
        ("kNN correlation", "knn_corr_k8", "er_knn8"),
    ]
    x = np.arange(len(pairs), dtype=float)
    width = 0.34

    structured_means = np.array([summary.loc[a, "mean"] for _, a, _ in pairs]) * 100.0
    random_means = np.array([summary.loc[b, "mean"] for _, _, b in pairs]) * 100.0
    structured_std = np.array([summary.loc[a, "std"] for _, a, _ in pairs]) * 100.0
    random_std = np.array([summary.loc[b, "std"] for _, _, b in pairs]) * 100.0

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(
        x - width / 2,
        structured_means,
        width,
        yerr=structured_std,
        capsize=4,
        label="Structured / data-driven graph",
    )
    ax.bar(
        x + width / 2,
        random_means,
        width,
        yerr=random_std,
        capsize=4,
        label="Random graph with the same density",
    )
    ax.set_xticks(x, [p[0] for p in pairs])
    ax.set_ylabel("Test accuracy, %")
    ax.set_title("The effect is not explained by edge density alone")
    ax.set_ylim(0, max(60, float(max(structured_means.max(), random_means.max()) + 6)))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_and_log(fig, "03_density_matched_controls.png", experiment)

    return {
        "grid_vs_random_same_density_pp": float(structured_means[0] - random_means[0]),
        "knn_vs_random_same_density_pp": float(structured_means[1] - random_means[1]),
    }


def plot_structure_recovery(df: pd.DataFrame, experiment) -> dict[str, float]:
    n_max = int(df["n"].max())
    data = df[df["n"] == n_max].copy()
    graph_order = [
        g
        for g in ["tree", "grid", "smallworld", "community", "scalefree", "ring", "geometric"]
        if g in set(data["graph"])
    ]
    method_order = [m for m in ["corr_static", "partial_static", "partial_free"] if m in set(data["method"])]

    means = data.groupby(["graph", "method"])["f1"].mean().unstack()
    stds = data.groupby(["graph", "method"])["f1"].std().unstack().fillna(0)
    means = means.reindex(graph_order)
    stds = stds.reindex(graph_order)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(graph_order), dtype=float)
    width = 0.24
    offsets = np.linspace(-width, width, len(method_order))

    for offset, method in zip(offsets, method_order):
        ax.bar(
            x + offset,
            means[method].to_numpy(),
            width,
            yerr=stds[method].to_numpy(),
            capsize=3,
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xticks(x, graph_order, rotation=20)
    ax.set_ylabel("Recovered-edge F1")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Recovering the hidden graph from data (n={n_max})")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_and_log(fig, "04_structure_recovery_by_graph.png", experiment)

    return {
        f"structure_{method}_mean_f1_n{n_max}": float(data[data["method"] == method]["f1"].mean())
        for method in method_order
    }


def plot_structure_by_sample_size(df: pd.DataFrame, experiment) -> None:
    method_order = [m for m in ["corr_static", "partial_static", "partial_free"] if m in set(df["method"])]
    grouped = df.groupby(["n", "method"])["f1"].agg(["mean", "std"]).reset_index()

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for method in method_order:
        part = grouped[grouped["method"] == method].sort_values("n")
        ax.errorbar(
            part["n"],
            part["mean"],
            yerr=part["std"],
            marker="o",
            capsize=3,
            label=METHOD_LABELS.get(method, method),
        )

    ax.set_xlabel("Number of generated samples")
    ax.set_ylabel("Recovered-edge F1")
    ax.set_ylim(0, 1.08)
    ax.set_title("Graph recovery as more data becomes available")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_and_log(fig, "05_structure_vs_sample_size.png", experiment)


def plot_refine_trajectory(df: pd.DataFrame, experiment, metric: str, filename: str, ylabel: str, title: str) -> dict[str, float]:
    rule_order = [
        r
        for r in ["keep", "random_rewire", "corr_static", "partial_static", "partial_free"]
        if r in set(df["rule"])
    ]
    grouped = df.groupby(["round", "rule"])[metric].agg(["mean", "std", "count"]).reset_index()
    grouped["sem"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for rule in rule_order:
        part = grouped[grouped["rule"] == rule].sort_values("round")
        x = part["round"].to_numpy()
        mean = part["mean"].to_numpy()
        std = part["sem"].fillna(0).to_numpy()
        if metric == "test_acc":
            mean = mean * 100.0
            std = std * 100.0
        ax.plot(x, mean, label=RULE_LABELS.get(rule, rule))
        ax.fill_between(x, mean - std, mean + std, alpha=0.12)

    ax.set_xlabel("Refinement round")
    ax.set_ylabel(ylabel)
    ax.set_title(title + " (mean ± SEM)")
    ax.grid(alpha=0.25)
    ax.legend()
    if metric == "rec_f1":
        ax.set_ylim(0, 1.05)
    fig.tight_layout()
    save_and_log(fig, filename, experiment)

    final_round = int(df["round"].max())
    last = df[df["round"] == final_round]
    result = {}
    for rule in rule_order:
        value = float(last[last["rule"] == rule][metric].mean())
        if metric == "test_acc":
            value *= 100.0
        result[f"refine_final_{metric}_{rule}"] = value
    return result


def log_summary_metrics(experiment, metrics: dict[str, float]) -> None:
    if experiment is None:
        return
    experiment.log_metrics(metrics)


def main() -> None:
    args = parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    main_df, structure_df, refine_df = load_results()
    experiment = start_comet(args)

    try:
        metrics: dict[str, float] = {}

        plot_main_graph_accuracy(main_df, experiment)
        metrics["alignment_gap_pp"] = plot_alignment_comparison(main_df, experiment)
        metrics.update(plot_density_controls(main_df, experiment))
        metrics.update(plot_structure_recovery(structure_df, experiment))
        plot_structure_by_sample_size(structure_df, experiment)
        metrics.update(
            plot_refine_trajectory(
                refine_df,
                experiment,
                metric="test_acc",
                filename="06_refine_test_accuracy.png",
                ylabel="Test accuracy, %",
                title="Replacing the initial graph sharply improves classification",
            )
        )
        metrics.update(
            plot_refine_trajectory(
                refine_df,
                experiment,
                metric="rec_f1",
                filename="07_refine_graph_recovery.png",
                ylabel="Recovered-edge F1",
                title="How well the current graph matches the hidden true graph",
            )
        )

        log_summary_metrics(experiment, metrics)
        print(f"\nDone. Figures are in: {FIGURES_DIR}")
        if experiment is not None:
            print("All figures were also logged to Comet ML.")
    finally:
        if experiment is not None:
            experiment.end()


if __name__ == "__main__":
    main()
