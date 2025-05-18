import torch
import torch.nn.utils.prune as prune
from torch import nn
from copy import deepcopy


@torch.enable_grad()
def evaluate_sensitivity(model, dataloader, loss_function, device, eps=1e-10):
    """
    Evaluates the sensitivity of model weights to changes in input data.

        Calculates the average absolute gradient for each weight parameter in the model
        across a given dataset and normalizes it by the maximum gradient value
        (or a small epsilon value if the max is zero).

        Args:
            model: The PyTorch model to evaluate.
            dataloader:  The DataLoader providing input data and targets.
            loss_function: The loss function used for calculating gradients.
            device: The device (e.g., 'cuda' or 'cpu') on which to perform calculations.
            eps: A small value added to the denominator during normalization
                 to prevent division by zero.

        Returns:
            dict: A dictionary where keys are weight parameter names and values are their
                  normalized average absolute gradients (sensitivity scores).
    """
    sensitivity = {}
    model.to(device)
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)

        model.zero_grad()
        output = model(data).squeeze(dim=-1)
        loss = loss_function(output, target)
        loss.backward()

        for param_name, param in model.named_parameters():
            if "weight" in param_name:
                if param_name in sensitivity:
                    sensitivity[param_name] += param.grad.abs()
                else:
                    sensitivity[param_name] = param.grad.abs()

    for k, sens_tensor in sensitivity.items():
        sensitivity[k] = sens_tensor / torch.max(
            sens_tensor.max(), torch.tensor(eps, device=sens_tensor.device)
        )

    return sensitivity


def do_pruning(model, pruning_type="Random", **kwargs):
    """
    Prunes a given model based on the specified type and parameters.

        Args:
            model: The neural network model to prune.
            pruning_type: The type of pruning to apply ("L1", "Random", "Remove", or "SensitivityBased"). Defaults to "Random".
            kwargs: Additional keyword arguments for configuring the pruning process,
                including 'amount' (float, proportion of weights to prune),
                'logs' (bool, whether to print logging information), and
                'sensitivity' (dict, sensitivity values for SensitivityBased pruning).

        Returns:
            The pruned model.
    """
    prune_model = deepcopy(model)

    amount = kwargs.get("amount", 0.3)
    logs = kwargs.get("logs", False)
    sensitivity = kwargs.get("sensitivity", {})
    counter = list(sensitivity.keys())

    def apply_l1_pruning(module):
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)
            if logs:
                print(f"Применен L1 прунинг к слою: {module}")
                print(f"Маска:\n{module.weight_mask}\n")

    def apply_random_pruning(module):
        if isinstance(module, nn.Linear):
            prune.random_unstructured(module, name="weight", amount=amount)
            if logs:
                print(f"Применен Random прунинг к слою: {module}")
                print(f"Маска:\n{module.weight_mask}\n")

    def remove_pruning(module):
        if isinstance(module, nn.Linear) and hasattr(module, "weight_mask"):
            prune.remove(module, "weight")
            if logs:
                print(f"Удалена маска прунинга в слое: {module}")

    def zero_weights_by_mask(module):
        if isinstance(module, nn.Linear) and hasattr(module, "weight_mask"):
            with torch.no_grad():
                module.weight.data *= module.weight_mask
                if logs:
                    print(f"Обнулены веса по маске в слое: {module}")
                    print(f"Текущие веса:\n{module.weight.data}\n")

    def apply_sensitivity_pruning(module):
        nonlocal counter

        if isinstance(module, nn.Linear):
            param_name = counter.pop(0)
            if param_name in sensitivity:
                sens_tensor = sensitivity[param_name]

                flat_weights = module.weight.view(-1)
                flat_sensitivity = sens_tensor.view(-1)

                num_params_to_prune = int(amount * flat_sensitivity.numel())
                _, indices = torch.topk(
                    flat_sensitivity, k=num_params_to_prune, largest=False
                )

                mask = torch.ones_like(flat_weights)
                mask[indices] = 0

                prune.custom_from_mask(
                    module, name="weight", mask=mask.view_as(module.weight)
                )

                if logs:
                    print(f"Применен Sensitivity-based прунинг к слою: {module}")
                    print(f"Маска:\n{mask.view_as(module.weight)}\n")

    pruning_funcs = {
        "L1": apply_l1_pruning,
        "Random": apply_random_pruning,
        "Remove": remove_pruning,
        "SensitivityBased": apply_sensitivity_pruning,
    }
    apply_func = pruning_funcs.get(pruning_type)

    if apply_func:
        prune_model.apply(apply_func)
        prune_model.apply(zero_weights_by_mask)
        prune_model.apply(remove_pruning)

    return prune_model
