"""
Строит baseline-графики из результатов по нескольким seed.

Читает:
- results/cnn_baseline_seed_*.csv
- results/directional_grid8_seed_*.csv
- results/generic_gnn_seed_*.csv

Сохраняет:
- results/figures/baselines/01_cnn_vs_directional.png
- results/figures/baselines/02_generic_gnn_baselines.png
- results/figures/baselines/03_generic_random_vs_grid8_per_seed.png
- results/baseline_summary.csv

Если настроен COMET_API_KEY, те же картинки отправляются в Comet.
"""

import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
FIGURE_DIR = RESULTS_DIR / "figures" / "baselines"
SUMMARY_PATH = RESULTS_DIR / "baseline_summary.csv"


def get_seed(path: Path) -> int:
    match = re.search(r"_seed_(\d+)\.csv$", path.name)
    if match is None:
        raise ValueError(f"Не удалось извлечь seed из {path}")
    return int(match.group(1))


def read_single_model(pattern: str, label: str) -> pd.DataFrame:
    rows = []

    for path in sorted(RESULTS_DIR.glob(pattern)):
        df = pd.read_csv(path)

        if "test_acc" not in df.columns:
            continue

        values = df["test_acc"].dropna()
        if values.empty:
            continue

        rows.append({
            "model": label,
            "seed": get_seed(path),
            "test_acc": float(values.iloc[-1]),
        })

    return pd.DataFrame(rows)


def read_generic() -> pd.DataFrame:
    rows = []

    for path in sorted(RESULTS_DIR.glob("generic_gnn_seed_*.csv")):
        df = pd.read_csv(path)
        if not {"graph_type", "test_acc"}.issubset(df.columns):
            continue

        seed = get_seed(path)

        for _, row in df.iterrows():
            rows.append({
                "seed": seed,
                "graph_type": str(row["graph_type"]),
                "test_acc": float(row["test_acc"]),
            })

    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col)["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .fillna({"std": 0.0})
    )


def add_points(ax, x_positions, value_groups):
    for x, values in zip(x_positions, value_groups):
        values = np.asarray(values) * 100.0
        offsets = np.array([0.0]) if len(values) == 1 else np.linspace(-0.08, 0.08, len(values))
        ax.scatter(np.full(len(values), x) + offsets, values, s=32, zorder=3)


def plot_cnn_vs_directional(cnn: pd.DataFrame, directional: pd.DataFrame):
    if cnn.empty or directional.empty:
        print("skip 01: нет CNN или directional CSV")
        return None

    combined = pd.concat([cnn, directional], ignore_index=True)
    order = ["CNN", "Directional Grid-8 GNN"]
    agg = summary(combined, "model").set_index("model").loc[order]

    means = agg["mean"].to_numpy() * 100.0
    stds = agg["std"].to_numpy() * 100.0
    x = np.arange(2)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(x, means, yerr=stds, capsize=5)

    add_points(
        ax,
        x,
        [cnn["test_acc"].to_numpy(), directional["test_acc"].to_numpy()],
    )

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.15,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Test accuracy, %")
    ax.set_title("CNN vs directional Grid-8 GNN")
    ax.set_ylim(max(0, min(means - stds) - 2.0), 100.2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = FIGURE_DIR / "01_cnn_vs_directional.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_generic(generic: pd.DataFrame):
    if generic.empty:
        print("skip 02: нет generic CSV")
        return None

    order = ["empty", "random", "grid8"]
    labels = ["Empty", "Random", "Grid-8"]

    if not set(order).issubset(set(generic["graph_type"])):
        print("skip 02: нужны empty/random/grid8")
        return None

    agg = summary(generic, "graph_type").set_index("graph_type").loc[order]
    means = agg["mean"].to_numpy() * 100.0
    stds = agg["std"].to_numpy() * 100.0
    x = np.arange(3)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(x, means, yerr=stds, capsize=5)

    add_points(
        ax,
        x,
        [
            generic.loc[generic["graph_type"] == name, "test_acc"].to_numpy()
            for name in order
        ],
    )

    for bar, value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy, %")
    ax.set_title("Generic GNN baseline")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    path = FIGURE_DIR / "02_generic_gnn_baselines.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_random_vs_grid(generic: pd.DataFrame):
    if generic.empty:
        return None

    pivot = generic.pivot_table(
        index="seed",
        columns="graph_type",
        values="test_acc",
        aggfunc="first",
    )

    if not {"random", "grid8"}.issubset(pivot.columns):
        return None

    pivot = pivot.dropna(subset=["random", "grid8"])
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    for seed, row in pivot.iterrows():
        values = [row["random"] * 100.0, row["grid8"] * 100.0]
        ax.plot([0, 1], values, marker="o", alpha=0.7)
        ax.text(1.03, values[1], f"seed {seed}", va="center", fontsize=8)

    mean_values = [
        pivot["random"].mean() * 100.0,
        pivot["grid8"].mean() * 100.0,
    ]
    ax.plot([0, 1], mean_values, marker="D", linewidth=3, label="Mean")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random", "Grid-8"])
    ax.set_ylabel("Test accuracy, %")
    ax.set_title("Random vs Grid-8 for each seed")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    path = FIGURE_DIR / "03_generic_random_vs_grid8_per_seed.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def build_summary(cnn, directional, generic):
    rows = []

    for df, name in [
        (cnn, "CNN"),
        (directional, "Directional Grid-8 GNN"),
    ]:
        if not df.empty:
            rows.append({
                "group": "sanity_baseline",
                "name": name,
                "mean_test_acc": df["test_acc"].mean(),
                "std_test_acc": df["test_acc"].std(ddof=1) if len(df) > 1 else 0.0,
                "runs": len(df),
            })

    if not generic.empty:
        for graph_type, group in generic.groupby("graph_type"):
            rows.append({
                "group": "generic_gnn",
                "name": graph_type,
                "mean_test_acc": group["test_acc"].mean(),
                "std_test_acc": group["test_acc"].std(ddof=1) if len(group) > 1 else 0.0,
                "runs": len(group),
            })

    return pd.DataFrame(rows)


def log_to_comet(paths, baseline_summary):
    if not os.environ.get("COMET_API_KEY"):
        print("COMET_API_KEY не задан: Comet пропущен")
        return

    try:
        from comet_ml import start

        experiment = start(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            project_name=os.environ.get("COMET_PROJECT_NAME", "gnn-experiments"),
        )
        experiment.set_name("baseline_visualization")

        for path in paths:
            experiment.log_image(
                image_data=str(path),
                name=path.name,
            )

        for _, row in baseline_summary.iterrows():
            prefix = (
                f"{row['group']}_{row['name']}"
                .lower()
                .replace(" ", "_")
                .replace("-", "_")
            )
            experiment.log_metric(
                f"{prefix}_mean_test_acc",
                float(row["mean_test_acc"]),
            )
            experiment.log_metric(
                f"{prefix}_std_test_acc",
                float(row["std_test_acc"]),
            )

        experiment.end()

    except Exception as e:
        print(f"COMET WARNING: {e}")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    cnn = read_single_model("cnn_baseline_seed_*.csv", "CNN")
    directional = read_single_model(
        "directional_grid8_seed_*.csv",
        "Directional Grid-8 GNN",
    )
    generic = read_generic()

    print(
        f"Loaded: CNN={len(cnn)}, "
        f"Directional={len(directional)}, "
        f"Generic rows={len(generic)}"
    )

    paths = []
    for path in [
        plot_cnn_vs_directional(cnn, directional),
        plot_generic(generic),
        plot_random_vs_grid(generic),
    ]:
        if path is not None:
            paths.append(path)
            print(f"saved: {path}")

    baseline_summary = build_summary(cnn, directional, generic)
    baseline_summary.to_csv(SUMMARY_PATH, index=False)
    print(f"saved: {SUMMARY_PATH}")

    if not baseline_summary.empty:
        printable = baseline_summary.copy()
        printable["mean_test_acc"] *= 100.0
        printable["std_test_acc"] *= 100.0
        print("\nSummary, %:")
        print(printable.to_string(index=False))

    log_to_comet(paths, baseline_summary)


if __name__ == "__main__":
    main()
