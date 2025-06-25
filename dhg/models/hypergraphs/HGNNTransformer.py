import torch
import torch.nn as nn

from dhg.models.hypergraphs.TransformerLayer import TransformerLayer
from dhg.nn import HGNNConv
from dhg.structure.hypergraphs import Hypergraph


class HGNNTransformer(nn.Module):
    def __init__(self, in_channels: int, hid_channels: int, num_classes: int, num_heads: int, use_bn: bool = False, drop_rate: float = 0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HGNNConv(in_channels, hid_channels, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(TransformerLayer(hid_channels, hid_channels, num_heads, use_bn=use_bn, drop_rate=drop_rate))
        self.layers.append(HGNNConv(hid_channels, num_classes, use_bn=use_bn, is_last=True))

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            X = layer(X, hg)
        return X