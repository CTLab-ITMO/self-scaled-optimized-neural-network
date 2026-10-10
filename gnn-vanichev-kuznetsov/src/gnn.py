import torch
from torch import nn

CLASS_COUNT = 10
BATCH_SIZE = 1500
LEARNING_RATE = 1e-2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SmallGCN(nn.Module):
    def __init__(self, num_classes=10, vertex_count=784):
        super().__init__()
        self.num_classes = num_classes
        self.vertex_count = vertex_count
        # Для поэлементного умножения на adj
        self.w1 = nn.Parameter(torch.ones(vertex_count))
        self.w2 = nn.Parameter(torch.ones(num_classes))

    def forward(self, images, adjacency, *, return_logits=False):
        A = torch.as_tensor(adjacency, dtype=self.w1.dtype, device=self.w1.device)
        images = torch.as_tensor(images, dtype=self.w1.dtype, device=self.w1.device)

        # Шаг 1: обновляем реальные вершины через A и их веса.
        updated_vertices = (images @ A) * self.w1[:self.vertex_count]

        # Шаг 2: фиктивные вершины получают информацию о классах
        logits = updated_vertices.sum(dim=1, keepdim=True) * self.w2

        if return_logits:
            return logits  # CrossEntropyLoss принимает значения до softmax.
        return torch.softmax(logits, dim=1)


def train_gnn(adjacency, images, labels, epochs, seed=0):
    torch.manual_seed(seed)
    model = SmallGCN(vertex_count=len(adjacency)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32, device=DEVICE)
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    image_batches = images.split(BATCH_SIZE)
    label_batches = labels.split(BATCH_SIZE)
    for _ in range(epochs):
        for batch_images, batch_labels in zip(image_batches, label_batches):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(
                model(batch_images, adjacency, return_logits=True), batch_labels,
            )
            loss.backward()
            optimizer.step()
    return model


def accuracy(model, adjacency, images, labels):
    adjacency = torch.as_tensor(adjacency, dtype=torch.float32, device=DEVICE)
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    correct = 0
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            predictions = model(images[start : start + BATCH_SIZE], adjacency).argmax(1)
            correct += (predictions == labels[start : start + BATCH_SIZE]).sum().item()
    return correct / len(images)
