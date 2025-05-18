import numpy as np
import torch
from torch import nn
import time

from train import train_one_model, valid_model


def get_number_of_params(model: nn.Module) -> int:
    """
    Calculates the total number of parameters in a PyTorch model.

        Args:
            model: The PyTorch module (model) whose parameters are to be counted.

        Returns:
            int: The total number of parameters in the model.  Sparse parameters
                 are counted as their non-zero elements, while dense parameters
                 are counted by their total number of elements.
    """
    n_params = 0
    for name, param in model.named_parameters():
        if param.is_sparse:
            n_params += param._nnz()
        else:
            n_params += param.numel()
    return n_params


def get_train_time(model, train_loader, valid_loader, criterion, epochs):
    """
    Calculates the time taken to train a model.

        Args:
            model: The model to be trained.
            train_loader: DataLoader for training data.
            valid_loader: DataLoader for validation data.
            criterion: Loss function used during training.
            epochs: Number of epochs to train for.

        Returns:
            float: The time taken to train the model in seconds.
    """
    start_time = time.time()
    train_one_model(
        model, train_loader, valid_loader, criterion=criterion, epochs=epochs
    )
    return time.time() - start_time


def get_valid_time(model, valid_loader, criterion, device):
    """
    Calculates the time taken to perform validation.

        Args:
            model: The model being validated.
            valid_loader: Data loader for the validation set.
            criterion: Loss function used during validation.
            device: Device (e.g., 'cpu', 'cuda') on which validation is performed.

        Returns:
            float: The time taken for validation in seconds.
    """
    start_time = time.time()
    valid_model(model, valid_loader, criterion, device)
    return time.time() - start_time


def get_avg_inference_time(model, dataloader, device):
    """
    Calculates the average inference time of a model on a given dataset.

        Args:
            model: The PyTorch model to evaluate.
            dataloader:  The data loader providing batches of input data.
            device: The device (e.g., 'cuda', 'cpu') to run the model on.

        Returns:
            float: The average inference time in seconds.
    """
    measures = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            start_time = time.time()
            x, y = x.to(device), y.to(device)
            output = model(x).view(-1)
            measures.append(time.time() - start_time)
    return np.mean(measures)


def inference_time_over_params(
    model: nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device
) -> float:
    """
    Calculates the ratio of average inference time to the number of model parameters.

        Args:
            model: The PyTorch model to analyze.
            dataloader: DataLoader for providing input data.
            device:  The device (CPU or GPU) on which to perform inference.

        Returns:
            float: The ratio of average inference time to the number of parameters.
    """
    n_params = get_number_of_params(model)
    inference_time = get_avg_inference_time(model, dataloader, device)
    return inference_time / n_params


def inference_time_over_added_params(
    new_model: nn.Module,
    old_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """
    Calculates the ratio of average inference time to the number of added parameters.

        Args:
            new_model: The new model.
            old_model: The old model.
            dataloader: DataLoader for measuring inference time.
            device:  The device to run the models on (e.g., 'cuda' or 'cpu').

        Returns:
            float: The ratio of average inference time to the number of added parameters.
    """
    n_params = get_number_of_params(new_model) - get_number_of_params(old_model)
    inference_time = get_avg_inference_time(new_model, dataloader, device)
    return inference_time / n_params
