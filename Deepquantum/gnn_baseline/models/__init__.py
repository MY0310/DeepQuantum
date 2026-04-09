"""GNN baseline models."""

from .gnn_models import (
    GNNConfig,
    GNNFeatureExtractor,
    ClassicalGNNSystem,
    create_gnn_model
)

__all__ = [
    'GNNConfig',
    'GNNFeatureExtractor',
    'ClassicalGNNSystem',
    'create_gnn_model'
]
