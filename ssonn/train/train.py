import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, mean_squared_error
from ssonn.model.utils import *
from ssonn.metrics.train_metrics import *
from ssonn.metrics.edge_finder import *
import numpy as np

from torch.utils.data import DataLoader

def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer, criterion, device):
    t0 = time.time()
    model.train()

    train_loss = 0
    for i in range(len(model.fc0.embed_linears)):
        model.fc0.embed_linears[i].training = True

    for i, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_time = time.time() - t0
    return train_loss, train_time


def eval_one_epoch(model, criterion, val_loader, task_type, device):
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
    else:
        metric = mean_squared_error(all_targets, all_preds)

    return val_loss, metric


def edge_replacement_func_new_layer(
    model, layer, optim, choose_threshold, ef, experiment, fully_connected=False
):
    chosen_edges = ef.choose_edges_threshold(model, layer, choose_threshold)

    print("Chosen edges:", chosen_edges, len(chosen_edges[0]))
    layer.replace_many(*chosen_edges, fully_connected)
    return len(chosen_edges[0])


def edge_deletion_func_new_layer(model, layer, optim, masks, choose_threshold, ef):
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
    train_losses = []
    train_losses_expand = []
    val_losses_expand = []
    last_plateau = float('inf')
    last_avg_expand_gradients = 0
    last_avg_embed_gradients = 0
    last_action = None
    for epoch in range(hyperparams["num_epochs"]):
        # print(list(model.named_parameters()))

        train_loss, train_time, expanding_gradients, embed_gradients = train_one_epoch(
            model, optimizer, criterion, train_loader, epoch, device, experiment
        )
        val_loss, val_accuracy = eval_one_epoch(
            model, criterion, test_loader, hyperparams["task_type"], device
        )

        val_losses += [val_loss]
        train_losses += [train_loss]

        print(
            f"Epoch {epoch + 1}/{hyperparams['num_epochs']}, Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
        )

        recent_changes = None
        if epoch - replace_epoch[-1] > max(
            hyperparams["prune_after_epochs"],
            hyperparams["min_epochs_between_expansions"],
            hyperparams["plateau_window_size"],
        ):
            recent_changes = [
                abs(train_losses[i] - train_losses[i - 1])
                for i in range(-hyperparams["plateau_window_size"], 0)
            ]
            avg_change = sum(recent_changes) / hyperparams["plateau_window_size"]
            print('avg_change:', avg_change)
            
            # val_losses_expand += [val_losses[-1]]
            # if len(val_losses_expand) == 1 or val_losses_expand[-1] <= val_losses_expand[-2] - 0.01:
            print(last_plateau - train_losses[-1])
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
