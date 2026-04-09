"""
Classical Graph Neural Network Models for Q-GAD Baseline Comparison

This module implements classical GNN architectures that serve as strong baselines
to compare against the quantum GBS approach. All models are designed to process
the same input data (transaction graphs + tabular features) and produce
compatible outputs for fair comparison.

Implemented Architectures:
1. GCN: Graph Convolutional Network (Kipf & Welling, ICLR 2017)
2. GAT: Graph Attention Network (Veličković et al., ICLR 2018)
3. GraphSAGE: Sampling and Aggregation (Hamilton et al., NeurIPS 2017)
4. GIN: Graph Isomorphism Network (Xu et al., ICLR 2019)

Theoretical Comparison to GBS:
- GBS: Quantum sampling → Hafnian-based topology features
- GNN: Message passing → Node embeddings → Graph-level representation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class GNNConfig:
    """Configuration for GNN models."""
    # Graph architecture
    gnn_type: str = "gcn"  # 'gcn', 'gat', 'sage', 'gin'
    num_layers: int = 3
    hidden_dim: int = 64
    dropout: float = 0.1

    # Output dimensions (matching Q-GAD system)
    graph_feature_dim: int = 9  # Match GBS quantum_feature_dim
    classical_feature_dim: int = 10  # Tabular feature dimension

    # Fusion
    fusion_hidden_dims: List[int] = None
    num_classes: int = 2

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.fusion_hidden_dims is None:
            self.fusion_hidden_dims = [64, 32]


class GraphConvLayer(nn.Module):
    """
    Single graph convolution layer.

    Mathematical formulation:
    H^(l+1) = σ(D^(-1/2) A D^(-1/2) H^(l) W^(l))

    where:
    - A: Adjacency matrix
    - D: Degree matrix
    - H^(l): Node features at layer l
    - W^(l): Learnable weights
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_dim]
        """
        # Compute adjacency matrix (sparse)
        num_nodes = x.size(0)
        edge_index = edge_index.cpu()

        # Create sparse adjacency matrix
        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1)),
            (num_nodes, num_nodes)
        ).to(x.device)

        # Add self-loops
        adj = adj + torch.sparse_coo_tensor(
            torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]).to(x.device),
            torch.ones(num_nodes),
            (num_nodes, num_nodes)
        ).to(x.device)

        # Compute degree normalization
        degree = torch.sparse.sum(adj, dim=1).to_dense()
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0

        # Normalized adjacency: D^(-1/2) A D^(-1/2)
        row, col = edge_index
        norm = degree_inv_sqrt[row] * degree_inv_sqrt[col]

        # Message passing
        out = torch.zeros_like(x)
        for i, j in zip(row, col):
            out[j] += x[i] * norm[i]

        # Linear transformation
        out = self.linear(out)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out


class MultiHeadAttentionLayer(nn.Module):
    """
    Multi-head attention for GAT (simplified implementation).

    Mathematical formulation:
    α_ij = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))

    where:
    - α_ij: Attention coefficient
    - a: Learnable attention vector
    - W: Learnable transformation
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.out_dim = out_dim

        # Linear transformations for attention
        self.W = nn.Linear(in_dim, out_dim)
        self.a = nn.Linear(2 * out_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(out_dim)

    def forward(self, x, edge_index):
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Updated node features [num_nodes, out_dim]
        """
        num_nodes = x.size(0)

        # Linear transformation
        h = self.W(x)  # [N, out_dim]

        # Compute attention coefficients
        row, col = edge_index

        # Concatenate features for each edge
        h_concat = torch.cat([h[row], h[col]], dim=1)  # [E, 2*out_dim]
        edge_scores = self.a(h_concat).squeeze()  # [E]
        edge_scores = F.leaky_relu(edge_scores, negative_slope=0.2)

        # Normalize attention scores per source node
        # Group by source node and apply softmax
        unique_src = torch.unique(row)
        normalized_scores = torch.zeros_like(edge_scores)

        for src in unique_src:
            mask = (row == src)
            if mask.sum() > 0:
                scores_src = edge_scores[mask]
                normalized_scores[mask] = F.softmax(scores_src, dim=0)

        # Aggregate messages
        out = torch.zeros_like(h)
        for i, (src, tgt, score) in enumerate(zip(row, col, normalized_scores)):
            out[tgt] += score * h[src]

        # Add self-loops (nodes keep some of their own features)
        out = out + h

        # Normalization and activation
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)

        return out


class GNNFeatureExtractor(nn.Module):
    """
    GNN-based graph feature extractor (classical equivalent of GBS kernel).

    Architectures:
    - GCN: Simple spectral convolution
    - GAT: Attention-based aggregation
    - GraphSAGE: Mean/max pooling aggregation
    - GIN: Powerful injective aggregation
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.gnn_type = config.gnn_type.lower()

        # Input projection
        self.input_proj = nn.Linear(config.classical_feature_dim, config.hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(config.num_layers):
            in_dim = config.hidden_dim
            out_dim = config.hidden_dim

            if self.gnn_type == "gcn":
                self.convs.append(GraphConvLayer(in_dim, out_dim, config.dropout))
            elif self.gnn_type == "gat":
                self.convs.append(MultiHeadAttentionLayer(in_dim, out_dim, num_heads=4, dropout=config.dropout))
            elif self.gnn_type == "sage":
                # GraphSAGE: Mean aggregation
                self.convs.append(nn.Sequential(
                    nn.Linear(in_dim * 2, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout)
                ))
            elif self.gnn_type == "gin":
                # GIN: MLP + sum aggregation
                self.convs.append(nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim)
                ))

            self.batch_norms.append(nn.BatchNorm1d(out_dim))

        # Graph-level pooling (readout)
        self.readout = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.graph_feature_dim)
        )

    def forward(self, x, edge_index, batch=None):
        """
        Extract graph-level features using GNN.

        Args:
            x: Node features [num_nodes, feature_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Graph assignment [num_nodes] (for batched graphs)

        Returns:
            Graph features [batch_size, graph_feature_dim]
        """
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.config.dropout, training=self.training)

        # GNN layers
        for conv, bn in zip(self.convs, self.batch_norms):
            if self.gnn_type in ["gcn", "gat"]:
                x = conv(x, edge_index)
            elif self.gnn_type == "sage":
                # Mean aggregation
                row, col = edge_index
                num_nodes = x.size(0)

                # Aggregate neighbor features
                neighbor_sum = torch.zeros_like(x)
                neighbor_count = torch.zeros(num_nodes, 1).to(x.device)
                for i, j in zip(row, col):
                    neighbor_sum[j] += x[i]
                    neighbor_count[j] += 1

                neighbor_mean = neighbor_sum / (neighbor_count + 1e-6)
                x = torch.cat([x, neighbor_mean], dim=1)
                x = conv[0](x)
                for layer in conv[1:]:
                    x = layer(x)

            elif self.gnn_type == "gin":
                # Sum aggregation (injective)
                row, col = edge_index
                num_nodes = x.size(0)

                # Aggregate neighbor features
                neighbor_sum = torch.zeros_like(x)
                for i, j in zip(row, col):
                    neighbor_sum[j] += x[i]

                # Add self features
                x = x + neighbor_sum
                x = conv(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.config.dropout, training=self.training)

        # Graph-level pooling
        if batch is None:
            # Single graph: global mean pooling
            graph_features = x.mean(dim=0, keepdim=True)
        else:
            # Batch of graphs: mean pooling within each graph
            graph_features = torch.zeros(
                batch.max().item() + 1,
                x.size(1),
                device=x.device
            )
            for i in range(batch.max().item() + 1):
                mask = (batch == i)
                graph_features[i] = x[mask].mean(dim=0)

        # Readout
        graph_features = self.readout(graph_features)

        return graph_features


class ClassicalGNNSystem(nn.Module):
    """
    Complete classical GNN system for fraud detection.

    Architecture:
    1. GNN Feature Extractor: Processes graph structure → graph embeddings
    2. Classical Encoder: Processes tabular features → tabular embeddings
    3. Fusion Layer: Combines both modalities
    4. Classifier: Final prediction

    This mirrors the Q-GAD architecture:
    - GNN Extractor ≈ GBS Quantum Kernel
    - Classical Encoder ≈ Classical Feature Encoder
    - Fusion ≈ Hybrid Fusion Layer
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config

        # GNN feature extractor (replaces GBS kernel)
        self.gnn_extractor = GNNFeatureExtractor(config)

        # Classical tabular feature encoder
        self.classical_encoder = nn.Sequential(
            nn.Linear(config.classical_feature_dim, config.fusion_hidden_dims[0]),
            nn.BatchNorm1d(config.fusion_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dims[0], config.fusion_hidden_dims[1])
        )

        # Fusion classifier
        fusion_input_dim = config.graph_feature_dim + config.fusion_hidden_dims[1]
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dims[0]),
            nn.BatchNorm1d(config.fusion_hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden_dims[0], config.num_classes)
        )

    def forward(self, x, edge_index, batch, classical_features):
        """
        Forward pass of classical GNN system.

        Args:
            x: Node features [num_nodes, node_feature_dim]
            edge_index: Edge indices [2, num_edges]
            batch: Graph assignment [num_nodes]
            classical_features: Tabular features [batch_size, classical_feature_dim]

        Returns:
            logits: [batch_size, num_classes]
        """
        # Extract graph features using GNN
        graph_features = self.gnn_extractor(x, edge_index, batch)

        # Encode classical tabular features
        classical_encoded = self.classical_encoder(classical_features)

        # Fusion
        combined = torch.cat([graph_features, classical_encoded], dim=1)
        logits = self.fusion(combined)

        return logits

    def predict(self, x, edge_index, batch, classical_features):
        """Make predictions (for evaluation)."""
        with torch.no_grad():
            logits = self.forward(x, edge_index, batch, classical_features)
            probs = F.softmax(logits, dim=1)
            return probs


def create_gnn_model(gnn_type: str = "gcn", config: GNNConfig = None):
    """
    Factory function to create GNN models.

    Args:
        gnn_type: Type of GNN ('gcn', 'gat', 'sage', 'gin')
        config: GNNConfig object

    Returns:
        ClassicalGNNSystem instance

    Example:
        >>> model = create_gnn_model('gcn')
        >>> logits = model(x, edge_index, batch, classical_features)
    """
    if config is None:
        config = GNNConfig(gnn_type=gnn_type)
    else:
        config.gnn_type = gnn_type

    model = ClassicalGNNSystem(config)
    return model
