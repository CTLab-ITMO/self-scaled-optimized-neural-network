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
        self.w = nn.Parameter(torch.ones(vertex_count + num_classes))

    def forward(self, images, adjacency, *, return_logits=False):
        # A: исходная adjacency, форма (vertex_count, vertex_count).
        A = torch.as_tensor(adjacency, dtype=self.w.dtype, device=self.w.device) # (VC, VC)

        images = torch.as_tensor(images, dtype=self.w.dtype, device=self.w.device) # (B, VC)


        O = A.new_zeros((self.vertex_count, self.num_classes))
        I = A.new_ones((self.num_classes, self.vertex_count))
        E = torch.eye(self.num_classes, dtype=A.dtype, device=A.device)

        # A | 0
        # I | E
        A_new = torch.cat((torch.cat((A, O), dim=1), torch.cat((I, E), dim=1)), dim=0)
        A_new = A_new * self.w # (VC + num_classes, VC + num_classes)

        # К изображениям допишем 10 единичек
        images_padded = torch.cat((images, images.new_ones((images.shape[0], self.num_classes))), dim=1) # (B, VC + num_classes)
        image_embs = images_padded @ A_new

        # Последние num_classes
        logits = image_embs[:, -self.num_classes:]

        if return_logits:
            return logits  # CrossEntropyLoss принимает значения до softmax.
        return torch.softmax(logits, dim=1)

def normalize(adjacency):
    adjacency = torch.tensor(adjacency, device=DEVICE) + torch.eye(len(adjacency), device=DEVICE)
    inverse_sqrt_degree = adjacency.sum(1) ** -0.5
    return inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]


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
