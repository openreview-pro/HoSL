import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dhg.structure.hypergraphs import Hypergraph


class TransformerLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int, use_bn: bool = False, drop_rate: float = 0.5, is_last: bool = False):
        super().__init__()
        self.is_last = is_last
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        self.key_hyperedge = nn.Linear(out_channels, out_channels)  # K for hyperedge-node attention
        self.num_heads = num_heads

    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
        # Compute node features (multi-head attention)
        Q = self.query(X).view(-1, self.num_heads, X.shape[1] // self.num_heads)  # Shape: [N, num_heads, head_dim]
        K = self.key(X).view(-1, self.num_heads, X.shape[1] // self.num_heads)    # Shape: [N, num_heads, head_dim]
        V = self.value(X).view(-1, self.num_heads, X.shape[1] // self.num_heads)  # Shape: [N, num_heads, head_dim]

        # Compute hyperedge features
        Xe = hg.v2e_aggregation(X, aggr="mean")  # Shape: [E, out_channels]

        # Compute K for hyperedge-node attention (single-head attention)
        Ke = self.key_hyperedge(Xe)  # Shape: [E, out_channels]
        Qe = self.query(X)  # Shape: [N, out_channels]

        # Compute attention between nodes and nodes (multi-head attention)
        attention_nodes = F.softmax(torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(Q.shape[-1]), dim=-1)
        X_node = torch.matmul(attention_nodes, V).view(-1, X.shape[1])  # Shape: [N, out_channels]

        # Compute attention between hyperedges and nodes (single-head attention)
        attention_hyperedges = F.softmax(torch.matmul(Qe, Ke.transpose(-1, -2)) / math.sqrt(Q.shape[-1]), dim=-1)  # Shape: [N, E]
        X_hyperedge = torch.matmul(attention_hyperedges, Xe)  # Shape: [N, out_channels]

        # Combine the results from node-node and hyperedge-node attention
        X = X_node + X_hyperedge
        # X = X_hyperedge
        # X = X_node

        if not self.is_last:
            X = self.act(X)
            if self.bn is not None:
                X = self.bn(X)
            X = self.drop(X)
        return X