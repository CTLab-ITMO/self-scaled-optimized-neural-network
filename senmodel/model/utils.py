from copy import deepcopy
import torch
from torch import nn

from .model import ExpandingLinear


def dense_to_sparse(dense_tensor: torch.Tensor, device="cpu") -> torch.Tensor:
    """
    Converts a dense tensor to a sparse tensor, with the option to specify the device.

    Args:
        dense_tensor (torch.Tensor): Dense tensor to convert.
        device (str): Device where the sparse tensor should reside.

    Returns:
        torch.sparse_coo_tensor: Sparse representation of the dense tensor.
    """
    indices = dense_tensor.nonzero(as_tuple=True)
    values = dense_tensor[indices]
    indices = torch.stack(indices).to(device)

    sparse_tensor = torch.sparse_coo_tensor(
        indices, values, dense_tensor.size(), device=device
    )
    return sparse_tensor


def convert_dense_to_sparse_network(
    model: nn.Module, layers, device="cpu"
) -> nn.Module:
    """
    Converts dense layers in a model to sparse equivalents.

        This function creates a deep copy of the input model and replaces specified
        dense layers with their sparse counterparts using ExpandingLinear layers.

        Args:
            model: The neural network model to convert.
            layers: A list or iterable of modules (layers) within the model that
                should be converted to sparse representations.
            device: The device on which to perform the conversion ('cpu' or 'cuda').
                Defaults to 'cpu'.

        Returns:
            nn.Module: A deep copy of the input model with specified dense layers
                replaced by sparse ExpandingLinear layers.
    """
    new_model = deepcopy(model)

    for name, module in model.named_children():
        if module in layers:
            sparse_weight = dense_to_sparse(module.weight.data, device=device)
            sparse_bias = dense_to_sparse(module.bias.data, device=device)
            setattr(
                new_model,
                name,
                ExpandingLinear(sparse_weight, sparse_bias, device=device),
            )
    return new_model


def get_model_last_layer(model):
    """
    Returns the last linear layer of a PyTorch model.

        Iterates through the modules of the model in reverse order and returns the first
        linear (or ExpandingLinear) layer found.

        Args:
            model: The PyTorch model to analyze.

        Returns:
            The last linear or ExpandingLinear layer in the model, or None if no such layer is found.
    """
    for layer in reversed(list(model.modules())):
        if isinstance(layer, (nn.Linear, ExpandingLinear)):
            return layer
    return None


def freeze_all_but_last(model: nn.Module):
    """
    Freezes the gradients of all parameters in a model except for those in the last layer.

        Args:
            model: The PyTorch model to modify.

        Returns:
            None: This method modifies the model in-place and does not return a value.
    """
    last_layer_params = get_model_last_layer(model)
    len_choose = last_layer_params.count_replaces

    for param in model.parameters():
        if last_layer_params is not param:
            param.requires_grad_(False)

    # if isinstance(last_layer_params, ExpandingLinear):
    #     last_layer_params.freeze_embeds(len_choose)
    with torch.no_grad():
        for i in range(len(last_layer_params.embed_linears) - 1, 0, -1):
            A = last_layer_params.embed_linears[i].weight_indices
            A_norm = A.clone()
            A_norm[1, :] -= len_choose[len(last_layer_params.embed_linears) - i - 1]

            B = last_layer_params.embed_linears[i - 1].weight_indices

            last_layer_params.embed_linears[i - 1].weight_values.grad[
                ~torch.isin(B[0, :], A[1, :]).nonzero()
            ].zero_()

        for i in range(len(last_layer_params.weight_values) - len_choose[-1]):
            last_layer_params.weight_values.grad[i] = 0


def freeze_only_last(model: nn.Module, len_choose=0):
    """
    Freezes the embeddings of only the last layer in a model.

        Args:
            model: The neural network model to modify.
            len_choose:  Determines how many embedding vectors to keep unfrozen
                         in the last layer. Defaults to 0, freezing all but the first len_choose vectors.

        Returns:
            None: This method modifies the model in-place and does not return a value.
    """
    last_layer_params = get_model_last_layer(model)
    last_layer_params.freeze_embeds(len_choose)


def unfreeze_all(model: nn.Module):
    """
    Unfreezes all parameters of a given model.

        Iterates through the model's parameters and sets `requires_grad` to True for each.
        Additionally, if a parameter is an instance of ExpandingLinear, its embedded layers are unfrozen as well.

        Args:
            model: The PyTorch model whose parameters should be unfrozen.

        Returns:
            None
    """
    for param in model.parameters():
        param.requires_grad_(True)

        if isinstance(param, ExpandingLinear):
            param.unfreeze_embeds()


def freeze_model(model, num_trainable_layers: int = 1):
    """
    Freezes the weights of layers in a model except for the last few.

        This function iterates through the layers of the provided model and sets
        `requires_grad` to False for all parameters in layers before the specified
        number of trainable layers, effectively freezing their weights during training.

        Args:
            model: The PyTorch model to freeze.
            num_trainable_layers: The number of last layers to keep trainable.
                Defaults to 1.

        Returns:
            The modified model with frozen layers.
    """
    for i in range(len(list(model.children())) - num_trainable_layers):
        for param in list(model.children())[i].parameters():
            param.requires_grad = False


def print_layer_status(model):
    """
    Prints the status of each layer in a PyTorch model.

        Iterates through all named parameters in the given model and prints whether
        each layer's parameters require gradient calculation (i.e., are not frozen).

        Args:
            model: The PyTorch model to analyze.

        Returns:
            None
    """
    for name, param in model.named_parameters():
        print(f"Layer: {name}, frozen: {not param.requires_grad}")
