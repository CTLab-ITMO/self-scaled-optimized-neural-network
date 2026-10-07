import torch
from torch import nn

CLASS_COUNT = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallGCN(nn.Module):
    def __init__(self, vertex_count):
        super().__init__()
        self.W = nn.Linear(vertex_count, vertex_count // 4)
        self.classifier = nn.Linear(vertex_count // 4, CLASS_COUNT)

    def forward(self, images, adjacency):
        return self.classifier(torch.tanh(self.W(images @ adjacency)))


def normalize(adjacency):
    adjacency = torch.tensor(adjacency, device=DEVICE) + torch.eye(len(adjacency), device=DEVICE)
    inverse_sqrt_degree = adjacency.sum(1) ** -0.5
    return inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]


def train_gnn(adjacency, images, labels, epochs, seed=0, history=None):
    torch.manual_seed(seed)
    model = SmallGCN(len(adjacency)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    normalized = normalize(adjacency)
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    for _ in range(epochs):
        batch = torch.randperm(len(images), device=DEVICE)[:BATCH_SIZE]
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(images[batch], normalized), labels[batch])
        if history is not None:
            history.append(loss.item())
        loss.backward()
        optimizer.step()
    return model


def accuracy(model, adjacency, images, labels):
    normalized = normalize(adjacency)
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    correct = 0
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            predictions = model(images[start : start + BATCH_SIZE], normalized).argmax(1)
            correct += (predictions == labels[start : start + BATCH_SIZE]).sum().item()
    return correct / len(images)
