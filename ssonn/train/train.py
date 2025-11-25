import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error
from ssonn.metrics.nonlinearity_metrics import AbsGradientEdgeMetric
from ssonn.model.utils import *
import torch.optim as optim
from ssonn.metrics.train_metrics import *
from ssonn.metrics.edge_finder import *
# import wandb
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

def train_one_epoch(model, optimizer, criterion, train_loader, epoch, device, experiment=None, use_log_tool=True):
    """
    Trains the model for one epoch.

        Args:
            model: The neural network model to train.
            optimizer: The optimization algorithm used for training.
            criterion: The loss function used to compute the loss.
            train_loader: DataLoader for the training data.
            device:  The device (CPU or GPU) on which to perform the training.

        Returns:
            tuple: A tuple containing the average training loss and the time taken for the epoch.
    """
    t0 = time.time()
    model.train()

    train_loss = 0
    expanding_gradients = []
    embed_gradients = []

    for i in range(len(model.fc0.embed_linears)):
        model.fc0.embed_linears[i].training = True

    for i, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()

        expanding_gradients += [model.fc0.weight_values.grad.clone().detach().cpu()]
        embed_gradients_temp = []
        for i in range(len(model.fc0.embed_linears)):
            if model.fc0.embed_linears[i].weight_values.grad is not None:
                embed_gradients_temp += [model.fc0.embed_linears[i].weight_values.grad.clone().detach().cpu()]
            else:
                embed_gradients_temp += [0]
        embed_gradients += [embed_gradients_temp]

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_time = time.time() - t0

    if use_log_tool:
        log_gradient_values("expanding grads", expanding_gradients, experiment, epoch)
        log_gradient_histograms("expanding grads hist", expanding_gradients, experiment, epoch)
        # for i in range(len(model.fc0.embed_linears)): 
        #     log_gradient_values(f"embed grads {i}", [el[i] for el in embed_gradients], experiment, epoch)
        #     log_gradient_histograms(f"embed grads hist {i}", [el[i] for el in embed_gradients], experiment, epoch)
    
    # for name, param in model.named_parameters():
    #     if param.requires_grad and param.grad is not None:
    #         print(f"{name}: grad norm = {param.grad.norm()}")
    #     elif not param.requires_grad:
    #         print(f"{name}: frozen (no grad)")

    return train_loss, train_time, expanding_gradients, embed_gradients


def eval_one_epoch(model, criterion, val_loader, task_type, device):
    """
    Args:
        model (torch.nn.Module)
        criterion (torch.nn.Module)
        val_loader (torch.utils.data.DataLoader)
        task_type (str): 'regression' or 'classification'.

    Returns:
        tuple: A tuple containing:
            - val_loss (float): Average loss over the validation set.
            - val_accuracy (float) / MSE (float): Accuracy score for classification tasks,
                                   or MSE for regression tasks.
    """

    model.eval()
    val_loss = 0
    all_targets = []
    all_preds = []
    
    for i in range(len(model.fc0.embed_linears)):
        model.fc0.embed_linears[i].training = False

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            if task_type == "classification":
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = outputs

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    val_loss /= len(val_loader)

    if task_type == "classification":
        metric = accuracy_score(all_targets, all_preds)
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        print("-" * 30)
    else:
        metric = mean_squared_error(all_targets, all_preds)

    return val_loss, metric


def edge_replacement_func_new_layer(
    model, layer, optim, choose_threshold, ef, experiment, fully_connected=False
):
    """
    Replaces edges in a given layer based on a threshold and updates the optimizer.

        This function selects edges to replace within a specified layer using a provided
        edge selection function and threshold. It then replaces these edges and
        reinitializes the optimizer with parameters from the modified layer, ensuring
        correct optimization for the new network structure.

        Args:
            model: The overall model containing the layer to be modified.
            layer: The specific layer in which edges will be replaced.
            optim: The optimizer used for training the model.
            choose_threshold: The threshold value used by the edge selection function.
            ef: An object providing the edge selection functionality (e.g., choose_edges_threshold).
            fully_connected: A boolean indicating whether the layer is fully connected,
                             influencing the replacement process.

        Returns:
            int: The number of edges that were successfully replaced in the layer.
    """

    chosen_edges = ef.choose_edges_threshold(model, layer, choose_threshold)
    # if len(layer.embed_linears) == 0:
        # log_replaces(chosen_edges, experiment)

    print("Chosen edges:", chosen_edges, len(chosen_edges[0]))
    layer.replace_many(*chosen_edges, fully_connected)

    if len(chosen_edges[0]) > 0:
        last_lrs = [optim.param_groups[i]["lr"] for i in range(len(optim.param_groups))]


        # Добавляем достаточно learning rates для всех embed_linears
    # Если embed_linears больше, чем было param_groups, используем последний lr
        while len(last_lrs) < len(layer.embed_linears) + 1:
            last_lrs.append(last_lrs[-1])
    
        # last_lrs.append(last_lrs[-1])
        print(last_lrs)
        optim.param_groups.clear()
        optim.state.clear()
        optim.add_param_group({"params": layer.weight_values, "lr": last_lrs[0]})
        for i in range(len(layer.embed_linears)):
            optim.add_param_group({"params": layer.embed_linears[i].weight_values, "lr": last_lrs[i+1]})
    
    return len(chosen_edges[0])


def edge_deletion_func_new_layer(model, layer, optim, masks, choose_threshold, ef):
    """
    Deletes edges from a given layer based on calculated metrics and thresholds.

        Args:
            model: The model containing the layer to modify.
            layer: The layer from which edges will be deleted.
            optim: The optimizer used for updating the model's parameters.
            masks: A tuple of masks indicating existing edge deletions.
            choose_threshold: The threshold value for determining edges to delete.
            ef: An object providing functions for calculating edge metrics.

        Returns:
            int: The total number of deleted edges (both embedding and expansion edges).
    """

    metr_edges_emb = ef.calculate_edge_metric_for_dataloader(
        model=model, layer=layer, to_normalise=False, embed=True
    )
    metr_edges_exp = ef.calculate_edge_metric_for_dataloader(
        model=model, layer=layer, to_normalise=False, embed=False
    )
    print(metr_edges_emb.shape, metr_edges_exp.shape)

    combined_metrics = torch.cat([metr_edges_emb, metr_edges_exp])
    print("combined_metrics", combined_metrics.shape)

    min_val = combined_metrics.min()
    max_val = combined_metrics.max()
    normalized = (combined_metrics - min_val) / (max_val - min_val + 1e-8)

    mask = normalized < choose_threshold
    print("mask", mask.shape)
    print(mask.sum())

    num_emb_edges = len(metr_edges_emb)
    print("num_emb_edges", num_emb_edges)
    mask_emb = mask[:num_emb_edges]
    mask_exp = mask[num_emb_edges:]

    mask_emb = ~masks[0] & ~mask_emb
    mask_exp = ~masks[1] & ~mask_exp

    print(mask_emb.sum(), mask_exp.sum())

    chosen_edges_emb = layer.embed_linears[-1].weight_indices[
        :, mask_emb.nonzero(as_tuple=True)[0]
    ]
    chosen_edges_exp = layer.weight_indices[:, mask_exp.nonzero(as_tuple=True)[0]]

    print("Chosen edges to del emb:", chosen_edges_emb, len(chosen_edges_emb[0]))
    print("Chosen edges to del exp:", chosen_edges_exp, len(chosen_edges_exp[0]))

    layer.delete_many(chosen_edges_emb, chosen_edges_exp)

    last_lrs = [optim.param_groups[i]["lr"] for i in range(len(optim.param_groups))]
    print(last_lrs)
    optim.param_groups.clear()
    optim.state.clear()

    optim.add_param_group({"params": layer.weight_values, 'lr': last_lrs[0]})
    for i in range(len(layer.embed_linears)):
        optim.add_param_group({"params": layer.embed_linears[i].weight_values, 'lr': last_lrs[i+1]})

    return len(chosen_edges_emb[0]) + len(chosen_edges_exp[0])


def log_gradient_image(gradients, experiment, epoch):
    avg_gradients = torch.stack(gradients).mean(dim=0)
    gradient_images = []
    for i in range(10):
        grad_img = avg_gradients[i*28*28:(i+1)*28*28].view(28, 28).numpy()
        gradient_images.append(grad_img)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, grad_img in enumerate(gradient_images):
        im = axes[i].imshow(grad_img, cmap='viridis')
        axes[i].set_title(f'Neuron {i} Gradients')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    experiment.log_figure(f"gradients_image", fig, step=epoch)
    plt.close(fig)
    
    # experiment.log_histogram_3d(avg_gradients, name="expanding linear grads", step=epoch)

def log_gradient_values(name, gradients, experiment, epoch):
    avg_gradients = torch.stack(gradients).mean(dim=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(range(len(avg_gradients)), avg_gradients, 'b-', alpha=0.7, linewidth=1)
    ax.scatter(range(len(avg_gradients)), avg_gradients, s=10, alpha=0.5, color='blue')
    
    ax.set_xlabel('Neuron Index (in flattened array)')
    ax.set_ylabel('Gradient Value')
    ax.set_title(f'Gradient Values vs Neuron Index (Epoch {epoch})')
    ax.grid(True, alpha=0.3)
    
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_ylim(-1e-3, 1e-3)
    
    plt.tight_layout()
    experiment.log_figure(name, fig, step=epoch)
    plt.close(fig)

def log_gradient_histograms(name, gradients, experiment, epoch):
    avg_gradients = torch.stack(gradients).mean(dim=0)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    ax.hist(avg_gradients, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_xlabel('Gradient Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)
    
    ax.set_xlim(-5e-4, 5e-4)
    plt.tight_layout()
    experiment.log_figure(name, fig, step=epoch)
    plt.close(fig)

def log_replaces(chosen_edges, experiment, ):
    image_indices = chosen_edges[0].cpu().numpy()
    pixel_indices = chosen_edges[1].cpu().numpy()
    
    class_masks = {}
    for class_idx in range(10):
        class_pixels = pixel_indices[image_indices == class_idx]
        
        mask = np.zeros(784)
        for pixel in class_pixels:
            mask[pixel] += 1
        
        mask = mask / mask.max()
        class_masks[class_idx] = mask.reshape(28, 28)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for class_idx in range(10):
        mask = class_masks.get(class_idx, np.zeros((28, 28)))
        
        im = axes[class_idx].imshow(mask, cmap='hot', interpolation='nearest')
        axes[class_idx].set_title(f'Class {class_idx}\n({np.sum(mask > 0)} pixels)')
        axes[class_idx].axis('off')
        plt.colorbar(im, ax=axes[class_idx], shrink=0.8)
    
    plt.tight_layout()
    experiment.log_figure("chosen_pixels_per_class", fig)
    plt.close(fig)


def train_sparse_recursive(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    hyperparams,
    device,
    experiment=None,
    use_log_tool=True
):
    """
    Trains a model with sparse recursive edge replacement and deletion.

        Args:
            model: The neural network model to train.
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            test_loader: DataLoader for the test dataset.
            criterion: The loss function to use.
            optimizer: The optimizer to use.
            hyperparams: A dictionary containing hyperparameters for training and sparsity.
            device: The device (CPU or GPU) to train on.
            experiment: 
            use_log_tool:
        Returns:
            None.  This method trains the model in-place and logs metrics using wandb.
    """

    model = model.to(device)

    ef = EdgeFinder(
        hyperparams["edge_importance_metric"],
        val_loader,
        device=device,
        aggregation_mode=hyperparams["edge_score_aggregation"],
        max_to_replace=hyperparams["max_new_edges_per_expansion"],
    )

    non_zero_masks = {}
    replace_epoch = [0]
    val_losses = []
    last_avg_abs_expand_gradients = 0
    last_avg_signed_expand_gradients = 0
    last_avg_embed_gradients = 0
    for epoch in range(hyperparams["num_epochs"]):
        # print(list(model.named_parameters()))

        train_loss, train_time, expanding_gradients, embed_gradients = train_one_epoch(
            model, optimizer, criterion, train_loader, epoch, device, experiment
        )
        val_loss, val_accuracy = eval_one_epoch(
            model, criterion, test_loader, hyperparams["task_type"], device
        )

        val_losses.append(val_loss)

        print(
            f"Epoch {epoch + 1}/{hyperparams['num_epochs']}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        if epoch - replace_epoch[-1] > max(
            hyperparams["prune_after_epochs"],
            hyperparams["min_epochs_between_expansions"],
            hyperparams["plateau_window_size"],
        ):
            recent_changes = [
                abs(val_losses[i] - val_losses[i - 1])
                for i in range(-hyperparams["plateau_window_size"], 0)
            ]
            avg_change = sum(recent_changes) / hyperparams["plateau_window_size"]
            if avg_change < hyperparams["plateau_threshold"]:
                len_choose = 0
                for layer_name in hyperparams["expansion_thresholds"].keys():
                    layer = model.__getattr__(layer_name)
                    len_choose += edge_replacement_func_new_layer(
                        model,
                        layer,
                        optimizer,
                        hyperparams["expansion_thresholds"][layer_name],
                        ef,
                        experiment,
                        hyperparams["start_fully_connected"],
                    )
                    # model.fc0.freeze_embeds(len_choose)
                    non_zero_masks[layer_name] = layer.get_non_zero_params()
                replace_epoch += [epoch]

        if (
            epoch - replace_epoch[-1] == hyperparams["prune_after_epochs"]
            and replace_epoch[-1] != 0
        ):
            len_choose = 0
            for layer_name in hyperparams["pruning_thresholds"].keys():
                layer = model.__getattr__(layer_name)
                len_choose += edge_deletion_func_new_layer(
                    model,
                    layer,
                    optimizer,
                    non_zero_masks[layer_name],
                    hyperparams["pruning_thresholds"][layer_name],
                    ef,
                ) 
            # print(list(model.named_parameters()))

        params_amount = get_params_amount(model)
        replace_params = 0
        for layer_name in hyperparams["expansion_thresholds"].keys():
            layer = model.__getattr__(layer_name)
            replace_params += len(
                ef.choose_edges_threshold(
                    model, layer, hyperparams["expansion_thresholds"][layer_name]
                )[0]
            )

        avg_abs_expand_gradients = torch.stack(expanding_gradients).abs().mean()
        avg_signed_expand_gradients = torch.stack(expanding_gradients).mean()
        # print(embed_gradients)
        # if embed_gradients[0] != []:
        #     avg_embed_gradients = torch.stack(embed_gradients).abs().mean()
        logs = {
            "val loss": val_loss,
            "val accuracy": val_accuracy,
            "train loss": train_loss,
            "params amount": params_amount,
            "params to replace amount": replace_params,
            "train time": train_time,
            "params ratio": (params_amount - replace_params) / params_amount,
            "lr": optimizer.param_groups[0]["lr"],
            "acc amount": val_accuracy / params_amount,
            "n_params over train_time": params_amount / train_time,
            "train_time over n_params": train_time / params_amount,
            "diff_mean_abs_expand_grads": avg_abs_expand_gradients - last_avg_abs_expand_gradients,
            "diff_mean_signed_expand_grads": avg_signed_expand_gradients - last_avg_signed_expand_gradients,
            "mean_abs_expand_grads": avg_abs_expand_gradients,
            "mean_signed_expand_grads": avg_signed_expand_gradients
        }
        last_avg_abs_expand_gradients = avg_abs_expand_gradients
        last_avg_signed_expand_gradients = avg_signed_expand_gradients
        # if embed_gradients[0] != []:
        #     logs["diff_mean_embed_grads"] = avg_embed_gradients - last_avg_embed_gradients
            # last_avg_embed_gradients = avg_embed_gradients

        if (epoch in replace_epoch) and epoch != 0:
            logs["len_choose"] = len_choose
        else:
            logs.pop("len_choose", None)

        if (epoch + hyperparams["prune_after_epochs"] in replace_epoch) and epoch != 0:
            logs["del_len_choose"] = len_choose
        else:
            logs.pop("del_len_choose", None)

        # wandb.log(logs)
        if use_log_tool:
            experiment.log_metrics(logs, epoch=epoch)
