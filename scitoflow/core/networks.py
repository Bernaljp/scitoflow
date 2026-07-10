"""
Neural network building blocks: MLP, GraphSAGE stack, activations.

FAITHFUL first-pass extraction from the method-development notebook
(Base/model1new_multivelo_organized.ipynb), Phase A1 consolidation.
Logic preserved verbatim; only module imports were added/normalized.
NOT yet unit-tested or made device-agnostic (hardcoded .cuda() remains) -
that is the research-software hardening pass. See PLAN.md Phase A.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

# --- faithful extraction: notebook cell 12 (model1new_multivelo_organized.ipynb) ---
# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================
from torch_geometric.nn import SAGEConv

class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, MLP_layers, activation='relu', bn=False, dropout=0.0, residual=False):
        super(MLP, self).__init__()
        self.activation = create_activation(activation)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.residual = residual
        
        self.layers = []
        self.layers.append(nn.Linear(input_dim, self.hidden_dim))
        if bn:
            self.layers.append(nn.BatchNorm1d(self.hidden_dim))
        self.layers.append(self.activation)
        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))
        
        for i in range(1, MLP_layers):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            if bn:
                self.layers.append(nn.BatchNorm1d(self.hidden_dim))
            self.layers.append(self.activation)
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(self.hidden_dim, output_dim))
        
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        residual = x
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.residual and isinstance(layer, nn.Linear):
                x += residual
                residual = x
        return x

class mySAGEConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, activation_fn='relu', batch_norm=False, dropout=0.0, residual=False):
        super(mySAGEConv, self).__init__()

        if n_layers <= 0:
            raise ValueError("Number of layers (n_layers) must be positive.")
        
        self.n_layers = n_layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.residual = residual

        self.activation = create_activation(activation_fn)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.convs = torch.nn.ModuleList()
        if self.batch_norm:
            self.bns = torch.nn.ModuleList()

        current_dim = in_channels
        for i in range(n_layers):
            if i < n_layers - 1:
                layer_out_channels = hidden_channels
            else:
                layer_out_channels = out_channels

            self.convs.append(SAGEConv(current_dim, layer_out_channels))
            if self.batch_norm:
                self.bns.append(nn.BatchNorm1d(layer_out_channels))
            current_dim = layer_out_channels

    def forward(self, x, edge_index):
        for i in range(self.n_layers):
            identity = x
            x_conv = self.convs[i](x, edge_index)
            if self.batch_norm:
                x_conv = self.bns[i](x_conv)
            if self.residual:
                if identity.shape[-1] == x_conv.shape[-1]:
                    x = x + x_conv
                else:
                    x = x_conv
            else:
                x = x_conv
            
            if i < self.n_layers - 1:
                x = self.activation(x)
            if self.dropout > 0:
                x = self.dropout_layer(x)
        return x

def create_activation(name):
    """Factory function for activation layers"""
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'selu':
        return nn.SELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")

