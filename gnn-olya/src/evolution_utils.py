import os
import random

try:
    from comet_ml import start as comet_start
except ImportError:
    comet_start = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import LineCollection


def edge_count(adj):
    return int(adj.sum().item() // 2)


def existing_edges(adj):
    return torch.nonzero(torch.triu(adj, diagonal=1), as_tuple=False)


def add_edges(adj, count, rng, max_edges):
    out = adj.clone()
    count = min(count, max_edges - edge_count(out))
    n = out.size(0)

    possible_edges = n * (n - 1) // 2
    current_edges = edge_count(out)

    if count < 0:
        raise ValueError("count must be non-negative")

    if current_edges + count > possible_edges:
        raise ValueError("Requested edge count exceeds graph capacity")

    while count > 0:
        u = rng.randrange(n)
        v = rng.randrange(n - 1)
        if v >= u:
            v += 1

        if out[u, v] == 0:
            out[u, v] = out[v, u] = 1.0
            count -= 1

    return out


def remove_edges(adj, count, rng):
    out = adj.clone()
    edges = existing_edges(out)
    count = min(count, len(edges))

    if count == 0:
        return out

    for idx in rng.sample(range(len(edges)), count):
        u, v = edges[idx].tolist()
        out[u, v] = out[v, u] = 0.0

    return out


# def rewire_edges(adj, count, rng, max_edges):
#     target_edges = edge_count(adj)
#     out = remove_edges(adj, count, rng)
#     return add_edges(out, target_edges - edge_count(out), rng, max_edges)


def start_comet_experiment(seed, parameters):
    if not os.environ.get("COMET_API_KEY") or comet_start is None:
        print("COMET WARNING: Comet недоступен")
        return None

    try:
        experiment = comet_start(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=os.environ.get("COMET_WORKSPACE"),
            project_name=os.environ.get("COMET_PROJECT_NAME", "gnn-experiments"),
        )
        experiment.set_name(f"evolution_variable_density_seed_{seed}")
        experiment.log_parameters(parameters)
        return experiment
    except Exception as e:
        print(f"COMET WARNING: не удалось запустить Comet: {e}")
        return None


def log_graph_figure(adj, name, experiment, figure_dir, step, val_acc=None):
    os.makedirs(figure_dir, exist_ok=True)

    edges = existing_edges(adj).cpu().numpy()
    node_ids = np.arange(784)
    x = node_ids % 28
    y = 27 - node_ids // 28

    fig, ax = plt.subplots(figsize=(7, 7))

    if len(edges):
        u, v = edges[:, 0], edges[:, 1]
        segments = np.stack(
            [
                np.column_stack([u % 28, 27 - u // 28]),
                np.column_stack([v % 28, 27 - v // 28]),
            ],
            axis=1,
        )
        ax.add_collection(LineCollection(segments, linewidths=0.5, alpha=0.3))

    ax.scatter(x, y, s=7, zorder=3)
    ax.set_aspect("equal")
    ax.set_xlim(-1, 28)
    ax.set_ylim(-1, 28)
    ax.axis("off")

    subtitle = f"edges={edge_count(adj)}"
    if val_acc is not None:
        subtitle += f", val_acc={val_acc * 100:.2f}%"
    ax.set_title(f"{name}\n{subtitle}")

    path = os.path.join(figure_dir, f"{name}.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    if experiment is not None:
        try:
            experiment.log_image(image_data=path, name=f"{name}.png", step=step)
            experiment.log_metric("edge_count", edge_count(adj), step=step)
            if val_acc is not None:
                experiment.log_metric("best_val_acc", val_acc, step=step)
        except Exception as e:
            print(f"COMET WARNING: не удалось залогировать {name}: {e}")


def save_checkpoint(checkpoint_path, history_path, generation, elites, rows):
    os.makedirs("results", exist_ok=True)

    torch.save(
        {
            "generation": generation,
            "elites": [
                {
                    "adj": elite["adj"].cpu(),
                    "fitness": float(elite["fitness"]),
                }
                for elite in elites
            ],
        },
        checkpoint_path,
    )
    pd.DataFrame(rows).to_csv(history_path, index=False)


def load_checkpoint(checkpoint_path, history_path):
    if not os.path.exists(checkpoint_path):
        return None

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    elites = [
        {
            "adj": item["adj"].float(),
            "fitness": float(item["fitness"]),
        }
        for item in checkpoint["elites"]
    ]

    rows = []
    if os.path.exists(history_path):
        rows = pd.read_csv(history_path).to_dict("records")

    return int(checkpoint["generation"]), elites, rows


def crossover_graphs(adj1, adj2, rng, max_edges):
    seed = rng.randint(0, 2**31 - 1)
    gen = torch.Generator().manual_seed(seed)
    
    mask = torch.rand(adj1.shape, generator=gen) < 0.5
    mask_triu = torch.triu(mask, diagonal=1)
    
    child_triu = torch.where(mask_triu, adj1, adj2)
    child_triu = torch.triu(child_triu, diagonal=1)
    
    out = child_triu + child_triu.T
    
    if edge_count(out) > max_edges:
        out = remove_edges(out, edge_count(out) - max_edges, rng)
    return out


def mutate_bitflip(adj, percent=0.05, rng=None, max_edges=None):
    rng = rng or random.Random()
    out = adj.clone()

    idx = torch.triu_indices(adj.size(0), adj.size(0), offset=1)
    values = out[idx[0], idx[1]]

    ones = torch.where(values == 1)[0]
    zeros = torch.where(values == 0)[0]

    k = max(1, round(len(ones) * percent))

    min_remove = 0
    if max_edges is not None:
        min_remove = max(
            0,
            (len(ones) + k - max_edges + 1) // 2,
        )
    min_remove = max(min_remove, k - len(zeros))

    max_remove = min(k, len(ones))
    remove_count = rng.randint(min_remove, max_remove)
    add_count = k - remove_count
    gen = torch.Generator().manual_seed(rng.randint(0, 2**31 - 1))

    remove_idx = ones[
        torch.randperm(len(ones), generator=gen)[:remove_count]
    ]
    add_idx = zeros[
        torch.randperm(len(zeros), generator=gen)[:add_count]
    ]

    flip_idx = torch.cat([remove_idx, add_idx])

    u = idx[0, flip_idx]
    v = idx[1, flip_idx]
    out[u, v] = 1.0 - out[u, v]
    out[v, u] = out[u, v]
    return out