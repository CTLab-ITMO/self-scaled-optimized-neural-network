import torch
from torch import nn


class MyModel(nn.Module):
    """
    A simple neural network model with customizable layer sizes and activation."""

    def __init__(self, all_sizes, activation=nn.ReLU):
        """
        Initializes the MyModel with specified layer sizes and activation function.

            Args:
                all_sizes: A tuple containing the input size, hidden layer sizes, and output size.
                activation: The activation function to use for hidden layers. Defaults to nn.ReLU.

            Returns:
                None
        """
        super(MyModel, self).__init__()
        input_size, *hidden_sizes, output_size = all_sizes
        layers = []
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(activation())
            current_size = hidden_size
        layers.append(nn.Linear(current_size, output_size))
        self.model = nn.Sequential(*layers)
        self.layers = all_sizes

    def forward(self, x):
        """
        Passes input through the model.

          Args:
            x: The input tensor.

          Returns:
            The output of the model after processing the input.
        """
        return self.model(x)


class SparseLinear(nn.Module):
    """
    A sparse linear layer that performs a matrix multiplication with sparsity."""

    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes a sparse linear layer.

            Args:
                in_features: The number of expected features in the input.
                out_features: The number of output features.
                bias: A boolean indicating whether to include a bias term. Defaults to True.

            Returns:
                None
        """
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features).to_sparse())
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, input):
        """
        Performs a sparse matrix multiplication followed by optional bias addition.

            Args:
                input: The input tensor to be multiplied with the weight matrix.

            Returns:
                The output tensor after performing the sparse matrix multiplication and adding the bias (if present).
        """
        output = torch.sparse.mm(self.weight, input.t()).t()
        if self.bias is not None:
            output += self.bias
        return output


def convert_sparse_weights_to_dense(model):
    """
    Converts sparse weights in a model to dense weights.

        Iterates through the modules of the given model and converts any
        SparseLinear layers' weights from a sparse format to a dense format,
        replacing the original sparse weight tensor with its dense equivalent.

        Args:
            model: The PyTorch model whose weights are to be converted.

        Returns:
            None: This method modifies the model in-place and does not return a value.
    """
    for module in model.modules():
        if isinstance(module, SparseLinear):
            module.weight = nn.Parameter(module.weight.to_dense())


def convert_dense_weights_to_sparse(model):
    """
    Converts dense weights in SparseLinear modules to sparse format.

        Iterates through the model's modules and converts the weight of any
        SparseLinear module from a dense tensor to a sparse tensor. This is done
        in-place, modifying the model directly.

        Args:
            model: The PyTorch model containing potentially dense weights in
                   SparseLinear layers.

        Returns:
            None: The function modifies the input model in place and does not return a value.
    """
    for module in model.modules():
        if isinstance(module, SparseLinear):
            module.weight = nn.Parameter(module.weight.to_sparse())


def convert_layer(layer):
    """
    Converts a PyTorch layer to a sparse equivalent if applicable.

        Currently, only nn.Linear layers are converted to SparseLinear layers.
        Other layer types are returned unchanged.

        Args:
            layer: The PyTorch layer to convert.

        Returns:
            The potentially converted layer (SparseLinear or the original layer).
    """
    if isinstance(layer, nn.Linear):
        print(layer.bias)
        print(layer.bias is not None)
        sparse_layer = SparseLinear(
            layer.in_features, layer.out_features, bias=layer.bias is not None
        )
        sparse_layer.weight = nn.Parameter(layer.weight.to_sparse())
        if layer.bias is not None:
            sparse_layer.bias = nn.Parameter(layer.bias)
        return sparse_layer
    else:
        return layer


def convert_to_sparse(model):
    """
    Recursively converts dense layers to sparse layers within a model.

        This function iterates through the model's children, and if it encounters an
        nn.Sequential module, it recursively processes each layer within that sequence.
        Otherwise, it directly applies the conversion to the module itself.

        Args:
            model: The PyTorch model to convert.

        Returns:
            The modified model with sparse layers.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            for idx, layer in enumerate(module):
                module[idx] = convert_layer(layer)
        else:
            setattr(model, name, convert_layer(module))
    return model


# base linear block
class SparseLinear(nn.Module):
    """
    Sparse linear layer with MAIN weight and bias matrices.

    Args:
        weight (torch.sparse.FloatTensor): The sparse weight matrix.
        bias (torch.sparse.FloatTensor): The sparse bias vector.

    Methods:
        forward(input):
            Performs the forward pass of the layer.
            Args:
                input (torch.Tensor): Input tensor to the layer.
            Returns:
                torch.Tensor: Output tensor after applying the sparse linear transformation.
    """

    def __init__(
        self, weight: torch.sparse.FloatTensor, bias: torch.sparse.FloatTensor
    ):
        """
        Initializes the SparseLinear layer.

          Args:
            weight: The sparse weight tensor.
            bias: The sparse bias tensor.

          Returns:
            None
        """
        super(SparseLinear, self).__init__()

        # sparse weight
        self.weight_indices = weight.coalesce().indices()
        self.weight_values = nn.Parameter(weight.coalesce().values())
        self.weight_size = list(weight.coalesce().size())

        # sparse bias
        # todo: think about bias representation
        self.bias_indices = bias.coalesce().indices()
        self.bias_values = nn.Parameter(bias.coalesce().values())
        self.bias_size = list(bias.coalesce().size())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Performs a sparse matrix multiplication with learned weights and bias.

          Args:
            input: The input tensor to be multiplied by the sparse weight matrix.

          Returns:
            torch.Tensor: The output tensor after performing the sparse matrix
              multiplication and adding the bias.
        """
        # create real sparse weight and bias
        # weight in separated form needed for optimizer
        sparse_weight = torch.sparse.FloatTensor(
            self.weight_indices, self.weight_values, self.weight_size
        )
        sparse_bias = torch.sparse.FloatTensor(
            self.bias_indices, self.bias_values, self.bias_size
        ).to_dense()

        output = torch.sparse.mm(sparse_weight, input.t()).t()
        output += sparse_bias.unsqueeze(0)

        return output


def dense_to_sparse(dense_tensor: torch.Tensor) -> torch.sparse.FloatTensor:
    """
    Converts a dense tensor to a sparse tensor.

        Args:
            dense_tensor: The dense tensor to convert.

        Returns:
            A sparse tensor representation of the input dense tensor.
    """
    indices = dense_tensor.nonzero(as_tuple=True)
    values = dense_tensor[indices]
    indices = torch.stack(indices)

    sparse_tensor = torch.sparse.FloatTensor(indices, values, dense_tensor.size())
    return sparse_tensor


def convert_dense_to_sparse_network(model: nn.Module) -> nn.Module:
    """
    Converts a given dense neural network model to a sparse neural network model.

    This function recursively iterate through the given model and replaces all instances of
    `nn.Linear` layers with `SparseLinear` layers

    Args:
        model (nn.Module): The dense neural network model to be converted.

    Returns:
        nn.Module: A new neural network model with sparse layers.
    """
    if isinstance(model, MyModel):
        new_model = model.__class__(model.layers)
    else:
        new_model = model.__class__()

    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            sparse_weight = dense_to_sparse(module.weight.data)
            sparse_bias = dense_to_sparse(module.bias.data)

            setattr(new_model, name, SparseLinear(sparse_weight, sparse_bias))
        else:
            setattr(new_model, name, convert_dense_to_sparse_network(module))
    return new_model


class SparseRecursiveLinear(nn.Module):
    """
    Sparse recursive linear layer.

    Args:
        sparse_linear (nn.Module): The sparse MAIN linear layer.
        previous (SparseRecursiveLinear or None): The previous layer in the recursive chain.
        is_last (bool, optional): Flag indicating if this is the last layer. Default is False.

    Methods:
        replace(child, parent):
            Replace an edge between two nodes in the layer to new node and two edges.
            Updates weight of MAIN layer and self embed weights.

        forward(input):
            Forward pass through all previous layers and the MAIN layer.
    """

    def __init__(self, sparse_linear, previous, is_last=False):
        """
        Initializes a SparseRecursiveLinear module.

            Args:
                sparse_linear: The linear layer to use for the current level.
                previous: The previous SparseRecursiveLinear module in the recursion.
                is_last: A boolean indicating if this is the last layer in the recursion.

            Returns:
                None
        """
        super(SparseRecursiveLinear, self).__init__()
        self.sparse_linear = sparse_linear
        self.previous = previous
        self.is_last = is_last

        self.embed_weight_indeces = torch.empty(2, 0, dtype=torch.int)
        self.embed_weight_values = nn.Parameter(torch.empty(0))
        self.embed_weight_size = torch.tensor([0, self.sparse_linear.weight_size[1]])

        self.child_counter = 0

    def replace(self, child, parent):
        """
        Replaces a connection in the sparse linear layer with a new one.

            This method removes an existing edge between 'child' and 'parent' nodes
            in the main weight matrix and adds a new edge from 'child' to a newly
            created node (indexed by max_parent). It also updates the embedding weights
            to reflect this change.

            Args:
                child: The child node index.
                parent: The parent node index.

            Returns:
                None
        """
        # mask of edge to remove in MAIN weight
        matches = (self.sparse_linear.weight_indices[0] == child) & (
            self.sparse_linear.weight_indices[1] == parent
        )
        index_to_remove = matches.nonzero(as_tuple=True)[
            0
        ]  # index of edge to remove in MAIN weight

        self.sparse_linear.weight_indices = self.sparse_linear.weight_indices[
            :, torch.logical_not(matches)
        ]  # remove edge from MAIN weight by masking

        # concated input from embed weight will pass through last vertices in MAIN layer
        max_parent = (
            self.sparse_linear.weight_indices[1].max() + 1
        )  # increase number of nodes in "input" of MAIN layer
        self.sparse_linear.weight_indices = torch.cat(
            [
                self.sparse_linear.weight_indices,
                torch.tensor([[child, max_parent]]).t(),
            ],
            dim=1,
        )  # add new edge to MAIN weight

        value_to_remove = self.sparse_linear.weight_values[
            index_to_remove
        ]  # get value of deleted edge from MAIN weight
        self.sparse_linear.weight_values.data = self.sparse_linear.weight_values[
            self.sparse_linear.weight_values != value_to_remove
        ]  # remove value of deleted edge from MAIN value list
        # todo smart weight generation
        self.sparse_linear.weight_values.data = torch.cat(
            [self.sparse_linear.weight_values.data, torch.rand(1)]
        )  # add new random weight to end of MAIN value list

        self.sparse_linear.weight_size[
            1
        ] += 1  # increase number of nodes in "input" of MAIN layer

        # add new edge to embed weight
        # where self.child_counter is number of nodes in embed weight
        # and parent is number of input node
        self.embed_weight_indeces = torch.cat(
            [
                self.embed_weight_indeces,
                torch.tensor([[self.child_counter, parent]]).t(),
            ],
            dim=1,
        )
        # todo smart weight generation
        self.embed_weight_values.data = torch.cat(
            [self.embed_weight_values, torch.rand(1)]
        )  # add new random weight to end of embed value list
        self.embed_weight_size[0] += 1
        self.child_counter += 1

    def forward(self, input):
        """
        Passes the input through the embedding and main layers.

            Args:
                input: The input tensor to be processed.

            Returns:
                torch.Tensor: The output tensor after passing through the embedding
                              and potentially a linear layer if this is the last recursive layer.
        """
        # if previous layer exists pass input through prevuios layer
        if self.previous is not None:
            input = self.previous.forward(input)
        # else pass through self weight

        # create real sparse weight
        sparse_embed_weight = torch.sparse.FloatTensor(
            self.embed_weight_indeces,
            self.embed_weight_values,
            list(self.embed_weight_size),
        )
        # pass thourgh self weight
        output = torch.sparse.mm(sparse_embed_weight, input.t()).t()
        # concat output of embed weight and input
        input = torch.cat([input, output], dim=1)

        # pass through MAIN weight if it's last recursive layer
        if self.is_last:
            return self.sparse_linear(input)

        return input
