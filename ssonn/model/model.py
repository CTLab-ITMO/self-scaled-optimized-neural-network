from abc import abstractmethod, ABC
from jaxtyping import Float
from typing import List

import torch
from torch import nn


class SparseModule(ABC, nn.Module):
    """
    A module for representing and manipulating sparse neural networks.

    This class provides functionality to create, modify, and represent sparse
    connections within a neural network using a graph-like structure. It allows
    for adding edges with associated weights and constructing sparse tensors from
    this data.
    """

    def __init__(self, weight_size, device="cpu"):
        """
        Initializes a SparseModule.

            Args:
                weight_size: The desired size of the weight tensor.
                device: The device to allocate tensors on (e.g., 'cpu' or 'cuda').

            Returns:
                None
        """

        super(SparseModule, self).__init__()
        self.register_buffer(
            "weight_indices", torch.empty(2, 0, dtype=torch.int, device=device)
        )
        self.weight_values = nn.Parameter(torch.empty(0, device=device))
        self.register_buffer("weight_size", torch.tensor(weight_size, dtype=torch.long))
        self.device = device

        self.activation = nn.ReLU()

    def add_edge(self, child, parent, original_weight):
        """
        Adds a new edge to the graph with its associated weight.

            Args:
                child: The index of the child node.
                parent: The index of the parent node.
                original_weight: The initial weight value for the edge.

            Returns:
                None
        """

        new_edge = torch.tensor(
            [[child, parent]], dtype=torch.int, device=self.device
        ).t()
        self.weight_indices = torch.cat([self.weight_indices, new_edge], dim=1)
        new_weight = (
            torch.tensor(original_weight, device=self.device).unsqueeze(0)
            if not isinstance(original_weight, torch.Tensor)
            else original_weight.to(self.device)
        )
        self.weight_values.data = torch.cat([self.weight_values.data, new_weight])

    def create_sparse_tensor(self):
        """
        Creates a sparse tensor from the stored weight data.

          Args:
            None

          Returns:
            torch.sparse_coo_tensor: A sparse coo tensor constructed from
              weight_indices, weight_values, and weight_size, placed on the specified device.
        """

        return torch.sparse_coo_tensor(
            self.weight_indices,
            self.weight_values,
            tuple(self.weight_size.tolist()),
            device=self.device,
        )

    @abstractmethod
    def replace(self, child, parent):
        """
        Replaces a child node with a parent node in the tree.

          This method is an abstract method and should be implemented by subclasses
          to define how the replacement operation is performed.

          Args:
            child: The child node to be replaced.
            parent: The parent node that will replace the child.

          Returns:
            None
        """

        pass

    def epsilon(self):
        """
        Generates a small random number for use in numerical stability.

          Args:
            None

          Returns:
            float: A randomly generated float between 0 and 1e-8, used to prevent division by zero or other numerical issues.
        """

        return torch.rand(1, device=self.device) * 1e-8

    def replace_many(self, children, parents):
        """
        Replaces multiple child nodes with their corresponding parent nodes.

            Iterates through the provided lists of children and parents, replacing each
            child with its associated parent using the 'replace' method.  Assumes that
            the lengths of both lists are equal.

            Args:
                children: The list of child nodes to be replaced.
                parents: The list of parent nodes to replace the children with.

            Returns:
                None
        """

        for c, p in zip(children, parents):
            self.replace(c, p)


class EmbedLinear(SparseModule):
    """
    Embeds a linear layer with sparse embedding."""

    def __init__(self, weight_size, device="cpu"):
        """
        Initializes the EmbedLinear layer.

            Args:
                weight_size: The size of the embedding weight.
                device: The device to use for computation (e.g., 'cpu' or 'cuda').
                    Defaults to 'cpu'.

            Returns:
                None
        """

        super(EmbedLinear, self).__init__([0, weight_size], device=device)
        self.child_counter = 0
        self.device = device

    def replace(self, child, parent, original_weight=1.0):
        """
        Replaces a node in the graph with a new edge.

          Adds an edge from a newly created child node to the specified parent node,
          incrementing internal counters for tracking size and available child IDs.

          Args:
            child: The child node identifier (not used directly in this method).
            parent: The parent node identifier.
            original_weight: The initial weight of the new edge. Defaults to 1.0.

          Returns:
            None
        """

        self.add_edge(self.child_counter, parent, original_weight=original_weight)
        self.weight_size[0] += 1
        self.child_counter += 1

    def make_linear(self, children, parents):
        """
        Creates linear edges between children and parents with random weights.

            Adds edges to the graph based on provided parent-child relationships,
            first establishing direct connections and then adding additional edges
            with random weights to ensure all children have a connection to each unique parent.

            Args:
                children: The tensor representing the child nodes.
                parents: The tensor representing the parent nodes.

            Returns:
                None
        """

        num_edges = len(children)
        done = [None] * num_edges

        for idx, (parent, child) in enumerate(zip(parents, range(num_edges))):
            self.add_edge(child, parent, original_weight=1)
            done[idx] = (child, parent.item())

        unique_parents = torch.unique(parents)
        for i in range(children.shape[0]):
            for j, parent in enumerate(unique_parents):
                if (i, parent.item()) not in done:
                    random_weight = self.epsilon()
                    self.add_edge(i, parent, original_weight=random_weight)
        self.weight_size[0] = children.shape[0]
        print(self.weight_size[0].item())

        #TODO make some experiments to decide
        # self.bias = nn.Parameter(torch.normal(mean=0, std=1, size=(self.weight_size[0].item(), )).to(self.device))
        self.bias = nn.Parameter(torch.zeros(self.weight_size[0], device=self.device))

    def forward(self, input):
        """
        Concatenates the input with a transformed version using sparse embedding.

          Args:
            input: The input tensor.

          Returns:
            A torch.Tensor: A tensor that is the concatenation of the original input
              and the activation of the result of a sparse matrix multiplication.
        """

        sparse_embed_weight = self.create_sparse_tensor()
        output = torch.sparse.mm(sparse_embed_weight, input.t()).t()
        output += self.bias.unsqueeze(0)
        return torch.cat([input, self.activation(output)], dim=1)

    def delete_many(self, children, parents):
        """
        Deletes specific edges from the weight indices and values.

            Args:
                children: The child nodes of the edges to be deleted.
                parents: The parent nodes of the edges to be deleted.

            Returns:
                None
        """

        to_delete = set(zip(children.tolist(), parents.tolist()))

        mask = torch.tensor(
            [
                (child.item(), parent.item()) not in to_delete
                for child, parent in zip(self.weight_indices[0], self.weight_indices[1])
            ],
            device=self.device,
        )

        self.weight_indices = self.weight_indices[:, mask]
        with torch.no_grad():
            self.weight_values.data = self.weight_values.data[mask]


class ExpandingLinear(SparseModule):
    """
    ExpandingLinear class for sparse linear transformations."""

    def __init__(
        self,
        weight: torch.sparse_coo_tensor,
        bias: torch.sparse_coo_tensor,
        device="cpu",
    ):
        """
        Initializes the ExpandingLinear module.

            Args:
                weight: The initial sparse weight tensor.
                bias: The initial sparse bias tensor.
                device: The device to place the tensors on (default 'cpu').

            Returns:
                None
        """

        super(ExpandingLinear, self).__init__(weight.size(), device=device)

        weight = weight.coalesce()
        self.weight_indices = weight.indices().to(device)
        self.weight_values = nn.Parameter(weight.values().to(device))

        self.embed_linears = nn.ModuleList()

        self.register_buffer(
            "count_replaces",
            torch.tensor([self.weight_indices.size(1)], dtype=torch.long),
        )

        bias = bias.coalesce()
        self.bias_indices = bias.indices().to(device)
        self.bias_values = nn.Parameter(bias.values().to(device))
        self.bias_size = list(bias.size())

        self.register_buffer("current_iteration", torch.tensor(-1, dtype=torch.long))
        self.device = device

    def replace(self, child, parent):
        """
        Replaces an existing weight with a new edge and updates the embedding.

            This method replaces a specific weight in the knowledge graph with a new
            edge, effectively updating the relationships between entities. It also
            handles expanding the embedding layer if necessary.

            Args:
                child: The child node of the weight to be replaced.
                parent: The parent node of the weight to be replaced.

            Returns:
                None
        """

        if self.current_iteration.item() == -1:
            self.current_iteration = 0

        if len(self.embed_linears) <= self.current_iteration.item():
            self.embed_linears.append(
                EmbedLinear(self.weight_size[1], device=self.device)
            )

        matches = (self.weight_indices[0] == child) & (self.weight_indices[1] == parent)

        original_weight = self.weight_values[matches].item()
        self.weight_indices = self.weight_indices[:, ~matches]
        self.weight_values = nn.Parameter(self.weight_values[~matches])

        max_parent = self.weight_indices[1].max().item() + 1

        for ch in torch.unique(self.weight_indices[0]):
            random_weight = self.epsilon()
            w = random_weight if ch != child else original_weight
            self.add_edge(ch, max_parent, w)

        with torch.no_grad():
            self.weight_size[1] += 1
        # self.embed_linears[self.current_iteration].replace(child, parent)

    def replace_many(self, children, parents, fully_connected=False):
        """
        Replaces multiple nodes in the graph with new nodes.

            This method updates the graph by replacing a set of 'children' nodes with
            corresponding 'parents' nodes. It also tracks the number of replacements
            made and potentially adjusts behavior based on whether a fully connected
            operation is requested.

            Args:
                children: The nodes to be replaced.
                parents: The nodes that will replace the children.
                fully_connected: A boolean flag indicating if the operation should be
                    treated as fully connected, potentially affecting parent selection.

            Returns:
                None: This method modifies the graph in place and does not return a value.
        """

        replaced_count = len(children)
        with torch.no_grad():
            self.count_replaces = torch.cat(
                [
                    self.count_replaces,
                    torch.tensor([replaced_count], device=self.device),
                ]
            )

        if len(children) and len(parents):
            self.current_iteration += 1

        if fully_connected:
            has_embed = len(self.embed_linears) != 0
            parents = torch.unique(
                self.embed_linears[-1].weight_indices[1]
                if has_embed
                else self.weight_indices[1]
            ).long()

        super().replace_many(children, parents)
        self.embed_linears[self.current_iteration.item()].make_linear(children, parents)

    def freeze_embeds(self, len_choose):
        if self.embed_linears:
            for i in range(len(self.embed_linears) - 1):
                for param in self.embed_linears[i].parameters():
                    param.requires_grad = False
        if hasattr(self, 'weight_values') and self.weight_values is not None:
            if len(self.weight_values) > len_choose:
                self.weight_values.requires_grad = False
                unfrozen_mask = torch.zeros_like(self.weight_values, dtype=torch.bool)
                unfrozen_mask[28*28*10-len_choose:] = True
                self.weight_values.requires_grad = True
                self.weight_values.register_hook(lambda grad: grad * unfrozen_mask)

    def unfreeze_embeds(self):
        """
        Unfreezes the embeddings in the model.

            This method sets `requires_grad` to True for all parameters within the
            `embed_linears` and, if present, `weight_values` attributes of the model.
            This allows these embedding weights to be updated during training.

            Parameters:
                None

            Returns:
                None
        """

        for embed_linear in self.embed_linears:
            for param in embed_linear.parameters():
                param.requires_grad = True

        if hasattr(self, "weight_values"):
            for param in self.weight_values:
                param.requires_grad = True

    def forward(self, input):
        """
        Applies the sparse linear transformation to the input.

            This method first passes the input through any embedded linear layers,
            then performs a sparse matrix multiplication using pre-defined weight and bias tensors,
            optionally applying masks to both weights and biases.

            Args:
                input: The input tensor to be transformed.

            Returns:
                torch.Tensor: The output tensor after the sparse linear transformation.
        """

        for embed_linear in self.embed_linears:
            input = embed_linear(input)

        masked_weight_values = (
            self.weight_values * self.weight_mask
            if hasattr(self, "weight_mask")
            else self.weight_values
        )
        sparse_weight = torch.sparse_coo_tensor(
            self.weight_indices,
            masked_weight_values,
            tuple(self.weight_size.tolist()),
            device=self.device,
        )

        masked_bias_values = (
            self.bias_values * self.bias_mask
            if hasattr(self, "bias_mask")
            else self.bias_values
        )
        sparse_bias = torch.sparse_coo_tensor(
            self.bias_indices, masked_bias_values, self.bias_size, device=self.device
        ).to_dense()

        output = torch.sparse.mm(sparse_weight, input.t()).t()
        output += sparse_bias.unsqueeze(0)

        return output

    def delete_many(self, emb_pairs, exp_pairs):
        """
        Deletes multiple edges from the weight indices.

            Args:
                children: The child nodes of the edges to delete.
                parents: The parent nodes of the edges to delete.
                to_delete: A list of tuples representing the edges to delete. Each tuple contains a (child, parent) pair.

            Returns:
                None
        """

        self.embed_linears[-1].delete_many(*emb_pairs)
        children, parents = exp_pairs

        to_delete = set(zip(children.tolist(), parents.tolist()))
        mask = torch.tensor(
            [
                (child.item(), parent.item()) not in to_delete
                for child, parent in zip(self.weight_indices[0], self.weight_indices[1])
            ],
            device=self.device,
        )

        self.weight_indices = self.weight_indices[:, mask]
        with torch.no_grad():
            self.weight_values.data = self.weight_values.data[mask]

    def get_non_zero_params(self, epsilon=1e-7):
        """
        Returns masks indicating non-zero parameters.

            This method generates two boolean masks based on the absolute values of
            the weight values and a given epsilon threshold.  One mask identifies
            elements where the embedding weights are greater than epsilon in magnitude,
            and the other identifies elements where the expanding weights are greater
            than epsilon in magnitude.

            Args:
                epsilon: The threshold value for determining non-zero parameters.

            Returns:
                tuple[torch.Tensor, torch.Tensor]: A tuple containing two boolean tensors.
                    The first tensor (embed_weight_mask) indicates where embedding weights
                    are non-zero, and the second (expanding_weight_mask) indicates where
                    expanding weights are non-zero.
        """

        last_embed_linear = self.embed_linears[-1]
        embed_weight_mask = torch.abs(last_embed_linear.weight_values) > epsilon
        expanding_weight_mask = torch.abs(self.weight_values) > epsilon
        return embed_weight_mask, expanding_weight_mask


class ScaledNet(nn.Module):
    def __init__(self, weight: Float[torch.Tensor, "dim0 dim1"], bias: Float[torch.Tensor, "dim0"], device: str='cpu'):
        self.weight = weight
        self.bias = bias
        self.device = device

        self.sub_linears = nn.ModuleList()

    def put_sub_linear(self, children: Float[torch.Tensor, "dim0"], parents: Float[torch.Tensor, "dim0"]):
        """
        Args:
            children: nodes, which takes output from parent nodes
            parents: nodes, which gives output to children nodes
        """
        sub_linear_weight = torch.zeros(())
        sub_linear_bias = ...
        self.sub_linears.append(SubLinear(sub_linear_weight, sub_linear_bias))


    def forward(self,): ...


class SubLinear(nn.Module):
    def __init__(self, weight: Float[torch.Tensor, "dim0 dim1"], bias: Float[torch.Tensor, "dim0"], device: str='cpu'):
        self.weight = weight
        self.bias = bias
        self.device = device


    def forward(self): ...


    






