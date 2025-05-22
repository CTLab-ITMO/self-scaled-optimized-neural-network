from ssonn.model.utils import *

def get_params_amount(model):
    """
    Calculates the total number of parameters in a model.

        Iterates through the model's children, summing the parameter counts from
        ExpandingLinear and nn.Linear layers.

        Args:
            model: The PyTorch model to analyze.

        Returns:
            int: The total number of parameters in the model.
    """
    
    amount = 0
    for _, layer in model.named_children():
        if isinstance(layer, ExpandingLinear):
            for linear in layer.embed_linears:
                amount += linear.weight_values.shape[0]
            amount += layer.weight_values.shape[0]
        elif isinstance(layer, nn.Linear):
            amount += layer.in_features * layer.out_features
    return amount


def get_zero_params_amount(model, eps=1e-8):
    """
    Counts the number of parameters with absolute values less than a threshold.

        This method iterates through the layers of a model and counts the number of
        parameters in ExpandingLinear and nn.Linear layers that have absolute values
        less than the specified epsilon value.

        Args:
            model: The neural network model to analyze.
            eps:  The threshold below which parameters are considered "zero". Defaults to 1e-8.

        Returns:
            int: The total number of zeroed parameters in the model.
    """
    
    amount = 0
    for _, layer in model.named_children():
        if isinstance(layer, ExpandingLinear):
            for linear in layer.embed_linears:
                amount += linear.weight_values[linear.weight_values.abs() < eps].shape[0]
            amount += layer.weight_values[layer.weight_values.abs() < eps].shape[0]
        elif isinstance(layer, nn.Linear):
            amount += layer.weight[layer.weight.abs() < eps].numel()
    return amount


def get_to_replace_params_amount(ef, model, layers, mask, choose_threshold):
    """
    Calculates the total number of parameters to be replaced based on edge selection.

        Args:
            ef: An object with a `choose_edges_threshold` method.
            model: The neural network model.
            layers: A list of layers in the model.
            mask: A mask used for filtering edges.
            choose_threshold: The threshold value for edge selection.

        Returns:
            int: The total number of parameters (edges) selected for replacement.
    """
    
    chosen_edges = 0
    for layer in layers:
        chosen_edges += len(ef.choose_edges_threshold(model, layer, choose_threshold, mask)[0])
    return chosen_edges