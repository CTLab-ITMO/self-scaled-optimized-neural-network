import numpy as np


def all_edges(vertex_count):
    rows, columns = np.triu_indices(vertex_count, k=1)
    return np.stack([rows, columns], axis=1)


def random_graph(edges, edge_count, vertex_count, rng):
    chosen = rng.permutation(edges)[:edge_count]
    adjacency = np.zeros((vertex_count, vertex_count), dtype=bool)
    adjacency[chosen[:, 0], chosen[:, 1]] = True
    adjacency[chosen[:, 1], chosen[:, 0]] = True
    return adjacency


def generate_graphs(graph_count, edge_counts, vertex_count, rng):
    edges = all_edges(vertex_count)
    graphs = []
    for index in range(graph_count):
        graphs.append(random_graph(edges, edge_counts[index], vertex_count, rng))
    return graphs
