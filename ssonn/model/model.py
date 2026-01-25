import torch
from torch import nn
import torch.nn.functional as F


class ScaledNet(nn.Module):
    """
    Implements the Self-Scaled Optimized Neural Network core module.
    """
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, device: str):
        """
        Args:
            weight (Tensor): Initial weight matrix of shape (out_features, in_features).
            bias (Tensor): Initial bias vector of shape (out_features).
            device (str): Computation device ('cpu' or 'cuda').
        """
        super(ScaledNet, self).__init__()
        self.device = device

        self.weight = nn.Parameter(weight.to(device))
        self.bias = nn.Parameter(bias.to(device))

        self.out_features, self.in_features = weight.shape

        self.sub_linears = nn.ModuleList()

    def put_sub_linear(self, parents: torch.Tensor, children: torch.Tensor):
        """
        Performs network expansion by inserting a new sub-layer and restructuring weights.

        1. Adds a new SubLinear layer to generate intermediate features.
        2. Expands the main weight matrix to accommodate these new features.
        3. Transfers weights from the old direct connections to the new paths.
        4. Prunes (zeros out) the original direct connections.

        Args:
            parents (Tensor): Indices of input neurons triggering expansion.
            children (Tensor): Indices of output neurons affected by expansion.
        """
        parents = parents.to(self.device).long()
        children = children.to(self.device).long()
        chosen_dim = parents.shape[0]

        self.sub_linears.append(SubLinear(parents, self.in_features, self.device))

        old_weight_data = self.weight.data
        weight_cat = torch.zeros((self.out_features, chosen_dim), device=self.device)
        weight_cat[children, torch.arange(0, chosen_dim)] = old_weight_data[children, parents]
        old_weight_data[children, parents] = 0.
        new_weight_data = torch.cat([old_weight_data, weight_cat], dim=1)

        self.weight = nn.Parameter(new_weight_data)

        self.in_features += chosen_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input 'x' grows in size as it passes through each SubLinear layer.
        """
        for sub_linear in self.sub_linears:
            x = sub_linear(x)

        return F.linear(x, self.weight, self.bias)


class SubLinear(nn.Module):
    """
    Represents an intermediate expansion layer.
    """
    def __init__(self, parents: torch.Tensor, input_dim: int, device: str):
        """
        Args:
            parents (Tensor): Indices of parent neurons to connect to.
            input_dim (int): Current size of the input feature vector.
        """
        super(SubLinear, self).__init__()
        self.device = device
        output_dim = parents.shape[0]

        weight_data = torch.empty((output_dim, input_dim), device=device).uniform_(0, 1e-8)
        weight_data[torch.arange(output_dim, device=device), parents] = 1.0
        self.weight = nn.Parameter(weight_data)

        self.bias = nn.Parameter(torch.zeros((1, output_dim), device=device))

        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        out = self.act(out)
        return torch.cat([x, out], dim=-1)
