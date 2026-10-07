import numpy as np
from sklearn.model_selection import train_test_split


class Evolution:
    def __init__(
        self, graphs, fitness_function, mutation_rate, seed=0,
        *, images, labels, train_size, test_size,
    ):
        self.fitness_function = fitness_function
        self.train_size = train_size
        self.test_size = test_size
        self.graph_count = len(graphs)
        vertex_count = len(graphs[0])
        self.mutation_probability = mutation_rate / vertex_count**2
        self.rng = np.random.default_rng(seed)
        self.split_rng = np.random.RandomState(seed)
        self.images, self.test_images, self.labels, self.test_labels = train_test_split(
            images,
            labels,
            test_size=test_size,
            random_state=self.split_rng,
            stratify=labels,
        )
        if self.train_size == len(self.images):
            self.train_images, self.train_labels = self.images, self.labels
        else:
            self.train_images, _, self.train_labels, _ = train_test_split(
                self.images,
                self.labels,
                train_size=self.train_size,
                random_state=self.split_rng,
                stratify=self.labels,
            )
        self.graphs = np.array(graphs)
        self.scores = self.evaluate(self.graphs)
        self.sort_by_score()

    def get_graphs(self):
        return self.graphs

    def get_scores(self):
        return self.scores

    def evaluate(self, graphs):
        """Use the same training and validation objects for every generation."""
        return np.array([
            self.fitness_function(
                graph, self.train_images, self.train_labels, self.test_images, self.test_labels,
            )
            for graph in graphs
        ])

    def evolve(self):
        survivor_count = self.graph_count // 2
        children = self.make_children(self.graphs[:survivor_count], self.graph_count - survivor_count)
        self.graphs[survivor_count:] = children
        # Survivors keep their scores on the same fixed validation set.
        self.scores[survivor_count:] = self.evaluate(children)
        self.sort_by_score()

    def sort_by_score(self):
        order = np.argsort(-self.scores)
        self.graphs = self.graphs[order]
        self.scores = self.scores[order]

    def make_children(self, parents, child_count):
        pairs = [self.rng.choice(len(parents), size=2, replace=False) for _ in range(child_count)]
        first, second = np.array(pairs).T
        return self.join(parents[first], parents[second])

    def join(self, first, second):
        return self.mutate(self.crossover(first, second))

    def crossover(self, first, second):
        take_first = self.rng.random(first.shape) < 0.5
        return symmetric(np.where(take_first, first, second))

    def mutate(self, graphs):
        flips = self.rng.random(graphs.shape) < self.mutation_probability
        return graphs ^ symmetric(flips)


def symmetric(matrices):
    upper = np.triu(matrices, k=1)
    return upper | np.swapaxes(upper, -1, -2)
