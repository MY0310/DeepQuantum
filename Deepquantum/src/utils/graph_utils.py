"""
Graph preprocessing utilities for Q-GAD system.

This module provides functions for:
- Ego-network extraction
- Graph matrix normalization
- Takagi-Autonne decomposition for quantum encoding
"""

import numpy as np
import scipy.linalg as la
import networkx as nx
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class SubgraphConfig:
    """Configuration for subgraph extraction."""
    max_nodes: int = 20  # Maximum nodes per subgraph (matches quantum modes)
    radius: float = 1.5  # Ego-network radius (1.5 or 2 hops)
    normalize: bool = True  # Normalize adjacency matrix
    fill_value: float = 0.0  # Value for padding
    sort_by: str = "pagerank"  # Sorting strategy: pagerank, degree, none


def extract_ego_network(
    graph: nx.Graph,
    center_node: int,
    config: SubgraphConfig
) -> nx.Graph:
    """
    Extract ego-network around a center node.

    Args:
        graph: Full transaction graph
        center_node: Target node to extract neighborhood from
        config: Subgraph configuration

    Returns:
        Induced subgraph around center node
    """
    # Extract k-hop neighborhood
    if config.radius == int(config.radius):
        # Integer radius
        subgraph = nx.ego_graph(graph, center_node, radius=int(config.radius))
    else:
        # For radius=1.5, extract 2-hop but focus on 1-hop core
        subgraph = nx.ego_graph(graph, center_node, radius=2)

        # Optional: weight 2-hop nodes lower
        for node in subgraph.nodes():
            if node not in graph.neighbors(center_node):
                # 2-hop nodes - could assign lower weights
                pass

    return subgraph


def normalize_adjacency(adj: np.ndarray, method: str = "symmetric") -> np.ndarray:
    """
    Normalize adjacency matrix for quantum encoding.

    Args:
        adj: Raw adjacency matrix
        method: Normalization method ('symmetric', 'row', 'max')

    Returns:
        Normalized adjacency matrix
    """
    if method == "symmetric":
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        deg = np.array(adj.sum(axis=1)).flatten()
        deg[deg == 0] = 1  # Avoid division by zero
        deg_inv_sqrt = np.power(deg, -0.5)
        D_inv_sqrt = np.diag(deg_inv_sqrt)
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt

    elif method == "row":
        # Row normalization
        deg = np.array(adj.sum(axis=1)).flatten()
        deg[deg == 0] = 1
        D_inv = np.diag(1.0 / deg)
        adj_norm = D_inv @ adj

    elif method == "max":
        # Scale by maximum value
        adj_norm = adj / (np.max(np.abs(adj)) + 1e-10)

    else:
        adj_norm = adj.copy()

    return adj_norm


def pad_or_truncate_subgraph(
    adj: np.ndarray,
    target_size: int,
    config: SubgraphConfig
) -> Tuple[np.ndarray, Dict]:
    """
    Pad or truncate subgraph adjacency to target size.

    Args:
        adj: Subgraph adjacency matrix
        target_size: Target matrix dimension
        config: Subgraph configuration

    Returns:
        Processed adjacency matrix and metadata
    """
    current_size = adj.shape[0]
    metadata = {"original_size": current_size, "padded": False, "truncated": False}

    if current_size == target_size:
        return adj, metadata

    if current_size < target_size:
        # Pad with zeros (vacuum modes)
        padded = np.zeros((target_size, target_size), dtype=adj.dtype)
        padded[:current_size, :current_size] = adj
        metadata["padded"] = True
        metadata["pad_size"] = target_size - current_size
        return padded, metadata

    else:
        # Need to truncate - sort nodes by importance
        metadata["truncated"] = True
        metadata["truncated_size"] = current_size - target_size

        # Compute node importance
        if config.sort_by == "pagerank":
            G_temp = nx.from_numpy_array(adj)
            importance = nx.pagerank(G_temp)
        elif config.sort_by == "degree":
            G_temp = nx.from_numpy_array(adj)
            importance = dict(G_temp.degree())
        else:
            importance = {i: i for i in range(current_size)}

        # Sort by importance (descending)
        sorted_nodes = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        keep_indices = [idx for idx, _ in sorted_nodes[:target_size]]

        # Reorder adjacency matrix
        adj_truncated = adj[np.ix_(keep_indices, keep_indices)]
        return adj_truncated, metadata


def takagi_autonne_decomposition(
    matrix: np.ndarray,
    tol: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Takagi-Autonne decomposition for symmetric matrix.

    For a complex symmetric matrix A, decomposes as:
        A = U @ diag(lambda) @ U.T

    where U is unitary and lambda contains singular values.

    Args:
        matrix: Symmetric matrix (adjacency or covariance)
        tol: Tolerance for numerical precision

    Returns:
        U (unitary matrix), lambda (singular values)

    Reference:
        Takagi, T. (1925). On an algebraic problem related to matric elements.
    """
    # Ensure symmetric
    if not np.allclose(matrix, matrix.T, atol=tol):
        raise ValueError("Matrix must be symmetric for Takagi decomposition")

    # Use SVD for symmetric matrices
    # For real symmetric matrices, this is equivalent to eigendecomposition
    try:
        # Try SVD first (more stable for complex matrices)
        U, s, _ = la.svd(matrix)

        # Takagi decomposition requires special handling for phases
        # For real symmetric matrices, SVD gives the decomposition directly
        lambda_vals = s

    except Exception as e:
        # Fallback to eigendecomposition
        lambda_vals, U = la.eigh(matrix)

        # Ensure non-negative eigenvalues for quantum encoding
        lambda_vals = np.maximum(lambda_vals, 0)

    return U.astype(np.complex128), lambda_vals.astype(np.complex128)


def matrix_to_squeezing_params(
    eigenvalues: np.ndarray,
    max_squeezing: float = 2.0,
    scale_factor: float = 0.9
) -> np.ndarray:
    """
    Convert eigenvalues to squeezing parameters.

    Relationship: tanh(r_i) = c * lambda_i

    Args:
        eigenvalues: Matrix eigenvalues from Takagi decomposition
        max_squeezing: Maximum squeezing parameter (in dB or natural units)
        scale_factor: Scaling factor c to ensure tanh(r) < 1

    Returns:
        Squeezing parameters r_i for each mode
    """
    # Normalize eigenvalues
    lambda_max = np.max(np.abs(eigenvalues))
    if lambda_max > 0:
        normalized_lambda = scale_factor * eigenvalues / lambda_max
    else:
        normalized_lambda = eigenvalues

    # Compute squeezing parameter: r = arctanh(c * lambda)
    # Clip to avoid numerical issues
    clipped_lambda = np.clip(normalized_lambda, -0.99, 0.99)
    squeezing_params = np.arctanh(clipped_lambda)

    # Apply maximum squeezing constraint
    squeezing_params = np.clip(squeezing_params, -max_squeezing, max_squeezing)

    return squeezing_params.astype(np.float32)


def preprocess_graph_for_quantum(
    graph: nx.Graph,
    center_node: int,
    config: SubgraphConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Complete pipeline: extract subgraph and prepare for quantum encoding.

    This is the main entry point for graph preprocessing.

    Args:
        graph: Full transaction graph
        center_node: Target node
        config: Subgraph configuration

    Returns:
        squeezing_params: Squeezing parameters for each mode
        unitary_matrix: Unitary matrix for interferometer
        adjacency_matrix: Processed adjacency matrix (for reference)
        metadata: Dictionary of processing info
    """
    # Step 1: Extract ego-network
    subgraph = extract_ego_network(graph, center_node, config)

    # Step 2: Convert to adjacency matrix
    adj = nx.to_numpy_array(subgraph)

    # Step 3: Normalize
    if config.normalize:
        adj = normalize_adjacency(adj, method="symmetric")

    # Step 4: Pad or truncate
    adj_processed, size_metadata = pad_or_truncate_subgraph(adj, config.max_nodes, config)

    # Step 5: Takagi-Autonne decomposition
    U, eigenvalues = takagi_autonne_decomposition(adj_processed)

    # Step 6: Convert to squeezing parameters
    squeezing_params = matrix_to_squeezing_params(eigenvalues)

    # Compile metadata
    metadata = {
        "center_node": center_node,
        "original_nodes": size_metadata["original_size"],
        "final_nodes": config.max_nodes,
        "padded": size_metadata["padded"],
        "truncated": size_metadata["truncated"],
        "eigenvalue_max": float(np.max(eigenvalues)),
        "eigenvalue_mean": float(np.mean(eigenvalues)),
        "squeezing_max": float(np.max(np.abs(squeezing_params))),
    }

    return squeezing_params, U, adj_processed, metadata


def compute_graph_density_features(adj: np.ndarray) -> Dict[str, float]:
    """
    Compute classical graph density features for comparison.

    Args:
        adj: Adjacency matrix

    Returns:
        Dictionary of density features
    """
    n = adj.shape[0]
    n_edges = np.sum(adj > 0)

    features = {
        "num_nodes": n,
        "num_edges": int(n_edges),
        "density": float(2 * n_edges / (n * (n - 1))) if n > 1 else 0.0,
        "avg_degree": float(np.mean(np.sum(adj > 0, axis=1))),
        "max_degree": int(np.max(np.sum(adj > 0, axis=1))),
        "clustering_coeff": float(nx.average_clustering(nx.from_numpy_array(adj))),
    }

    return features


if __name__ == "__main__":
    # Test the preprocessing pipeline
    print("Testing graph preprocessing utilities...")

    # Create a synthetic transaction graph
    G = nx.erdos_renyi_graph(50, 0.1, seed=42)

    # Add edge weights
    for u, v in G.edges():
        G[u][v]["weight"] = np.random.rand()

    config = SubgraphConfig(max_nodes=20, radius=1.5)

    # Test preprocessing on a few nodes
    test_nodes = list(G.nodes())[:3]

    for node in test_nodes:
        print(f"\n--- Processing node {node} ---")
        squeeze, U, adj, meta = preprocess_graph_for_quantum(G, node, config)

        print(f"Original nodes: {meta['original_nodes']}")
        print(f"Padded/Truncated: {meta['padded']}/{meta['truncated']}")
        print(f"Eigenvalue range: [{meta['eigenvalue_max']:.4f}]")
        print(f"Squeezing range: [{meta['squeezing_max']:.4f}]")

        # Compute classical features
        features = compute_graph_density_features(adj)
        print(f"Density: {features['density']:.4f}")
        print(f"Clustering: {features['clustering_coeff']:.4f}")

    print("\nGraph preprocessing test completed!")
