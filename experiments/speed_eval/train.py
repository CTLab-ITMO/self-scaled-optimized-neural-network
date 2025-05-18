import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from sparse_model import (
    convert_sparse_weights_to_dense,
    convert_dense_weights_to_sparse,
)


def train_one_model(
    model,
    train_loader,
    valid_loader,
    device,
    epochs=10000,
    patience=10,
    min_delta=1e-4,
    criterion=torch.nn.L1Loss(),
    lr=3e-4,
    optimizer=None,
):
    """
    Trains a given model for a specified number of epochs with early stopping.

        Args:
            model: The PyTorch model to train.
            train_loader: DataLoader for the training dataset.
            valid_loader: DataLoader for the validation dataset.
            device:  The device (e.g., 'cpu', 'cuda') on which to perform training.
            epochs: The maximum number of epochs to train for.
            patience: Number of epochs with no improvement before early stopping.
            min_delta: Minimum change in validation loss to qualify as an improvement.
            criterion: Loss function used for training.
            lr: Learning rate for the optimizer.
            optimizer: Optional optimizer instance; if None, Adam is used.

        Returns:
            float: The best validation loss achieved during training.
    """
    if not optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float("inf")
    epochs_no_improve = 0
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        model.train()

        convert_sparse_weights_to_dense(model)

        for data in train_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            x = x.to_dense() if x.is_sparse else x
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1), y)
            loss.backward()

            optimizer.step()

        convert_dense_weights_to_sparse(model)

        val_loss = valid_model(model, valid_loader, criterion, device=device)
        val_losses.append(val_loss)

        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    plt.figure(figsize=(10, 5))
    plt.plot(val_losses, label="Validation Loss")
    plt.axhline(
        y=sum(val_losses) / len(val_losses),
        color="r",
        linestyle="--",
        label="Average Loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Loss per Epoch")
    plt.legend()
    plt.show()

    return best_loss


def valid_model(model, valid_loader, criterion, device):
    """
    Calculates the mean squared error (MSE) loss on a validation dataset.

        Args:
            model: The model to validate.
            valid_loader: DataLoader for the validation dataset.
            criterion: Loss function used for evaluation.
            device: Device (e.g., 'cpu', 'cuda') where tensors are located.

        Returns:
            float: The average MSE loss over the validation dataset.
    """
    model.eval()
    mse = 0
    with torch.no_grad():
        for data in valid_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            output = model(x).view(-1)
            mse += criterion(output, y)
    return mse.item() / len(valid_loader)
