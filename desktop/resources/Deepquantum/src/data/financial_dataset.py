"""
Financial Graph Dataset for Q-GAD system.

This module implements PyTorch datasets for financial transaction graphs,
including support for Elliptic++ Bitcoin dataset and synthetic data generation.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import pickle


def inspect_quantum_cache(cache: Dict, max_nodes: int) -> Dict[str, float]:
    """
    Inspect cache quality to detect degenerate cached quantum parameters.
    """
    sq = np.asarray(cache.get("squeezing"))
    U = np.asarray(cache.get("unitary"))
    meta = cache.get("metadata", [])

    if sq.ndim != 2 or U.ndim != 3:
        return {"valid_shape": 0.0}

    n = float(sq.shape[0]) if sq.shape[0] > 0 else 1.0
    sq_row_norm = np.linalg.norm(sq, axis=1)
    sq_zero_ratio = float(np.mean(sq_row_norm == 0))

    eye = np.eye(max_nodes, dtype=U.dtype)
    max_abs_diff = np.max(np.abs(U - eye), axis=(1, 2))
    unitary_identity_ratio = float(np.mean(max_abs_diff < 1e-8))

    center_ratio = 0.0
    if isinstance(meta, list) and len(meta) > 0:
        center_count = sum(
            1 for m in meta if isinstance(m, dict) and ("center_node" in m)
        )
        center_ratio = float(center_count / max(1, len(meta)))

    return {
        "valid_shape": 1.0,
        "num_samples": float(sq.shape[0]),
        "sq_zero_ratio": sq_zero_ratio,
        "unitary_identity_ratio": unitary_identity_ratio,
        "center_ratio": center_ratio,
        "finite_ok": float(np.isfinite(sq).all() and np.isfinite(U).all()),
    }


def is_quantum_cache_usable(cache: Dict, max_nodes: int) -> bool:
    """
    Conservative cache validation.
    """
    try:
        required = {"squeezing", "unitary", "metadata"}
        if not required.issubset(set(cache.keys())):
            return False

        report = inspect_quantum_cache(cache, max_nodes=max_nodes)
        if report.get("valid_shape", 0.0) < 1.0:
            return False
        if report.get("finite_ok", 0.0) < 1.0:
            return False
        # Missing center_node mapping for a large fraction often indicates old fallback cache.
        if report.get("center_ratio", 0.0) < 0.95:
            return False
        # Fully degenerate cache (all-zero squeezing + all-identity unitary) is unusable.
        if report.get("sq_zero_ratio", 1.0) > 0.98 and report.get("unitary_identity_ratio", 1.0) > 0.98:
            return False
        return True
    except Exception:
        return False


class FinancialGraphDataset(Dataset):
    """
    PyTorch Dataset for financial transaction graphs with quantum encoding.

    This dataset:
    1. Loads transaction graph and labels
    2. Extracts ego-networks for each node
    3. Precomputes quantum encoding parameters (squeezing, unitary)
    4. Provides classical features

    For large graphs, precomputes and caches quantum parameters.
    """

    def __init__(
        self,
        edge_list: Union[str, pd.DataFrame, List[Tuple]],
        node_features: Optional[Union[str, pd.DataFrame, np.ndarray]] = None,
        labels: Optional[Union[str, pd.DataFrame, np.ndarray, Dict]] = None,
        max_nodes: int = 20,
        ego_radius: float = 1.5,
        cache_dir: Optional[str] = None,
        preload: bool = True,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.

        Args:
            edge_list: Path to CSV/DataFrame or list of (source, target, weight) edges
            node_features: Path to CSV/DataFrame or array of node features [n_nodes, n_features]
            labels: Path to CSV/DataFrame or dict/array of node labels (0=normal, 1=fraud)
            max_nodes: Maximum nodes per subgraph (quantum modes)
            ego_radius: Ego-network extraction radius
            cache_dir: Directory to cache preprocessed data
            preload: Whether to preload all data into memory
            transform: Optional transform function
        """
        super().__init__()
        self.max_nodes = max_nodes
        self.ego_radius = ego_radius
        self.transform = transform
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.preload = preload

        # Load graph
        self.graph = self._load_graph(edge_list)

        # Load node features
        self.node_features = self._load_node_features(node_features)

        # Load labels
        self.labels = self._load_labels(labels)

        # Get node list
        self.nodes = list(self.graph.nodes())
        self.num_nodes = len(self.nodes)

        # Create node to index mapping
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        # Precompute quantum parameters (or load from cache)
        self.quantum_params = self._compute_or_load_quantum_params()
        self.cache_node_to_idx = {}
        for idx, meta in enumerate(self.quantum_params.get("metadata", [])):
            if isinstance(meta, dict) and "center_node" in meta:
                try:
                    self.cache_node_to_idx[int(meta["center_node"])] = idx
                except (TypeError, ValueError):
                    continue
        self._ondemand_quantum_cache = {}

        print(f"Dataset initialized: {self.num_nodes} nodes, "
              f"{self.graph.number_of_edges()} edges")

    def _load_graph(self, edge_list: Union[str, pd.DataFrame, List, nx.Graph]) -> nx.Graph:
        """Load transaction graph from various formats."""
        if isinstance(edge_list, nx.Graph):
            # Already a NetworkX graph
            G = edge_list
        elif isinstance(edge_list, str):
            # Load from file
            if edge_list.endswith(".csv"):
                df = pd.read_csv(edge_list)
                if df.shape[1] >= 2:
                    # Assume columns: source, target, [weight]
                    cols = df.columns.tolist()
                    G = nx.from_pandas_edgelist(
                        df,
                        source=cols[0],
                        target=cols[1],
                        edge_attr=cols[2:] if len(cols) > 2 else None
                    )
                else:
                    raise ValueError("Edge list CSV must have at least 2 columns")
            elif edge_list.endswith(".edgelist"):
                G = nx.read_edgelist(edge_list)
            elif edge_list.endswith(".pkl"):
                with open(edge_list, "rb") as f:
                    G = pickle.load(f)
            else:
                raise ValueError(f"Unsupported edge list format: {edge_list}")

        elif isinstance(edge_list, pd.DataFrame):
            G = nx.from_pandas_edgelist(edge_list)

        elif isinstance(edge_list, list):
            if len(edge_list[0]) >= 2:
                G = nx.from_edgelist(edge_list)
            else:
                raise ValueError("Edges must have at least (source, target)")

        else:
            raise TypeError(f"Unsupported edge_list type: {type(edge_list)}")

        # Ensure graph is undirected (or make it undirected)
        if not G.is_directed():
            G = G.to_undirected()

        return G

    def _load_node_features(
        self,
        node_features: Optional[Union[str, pd.DataFrame, np.ndarray]]
    ) -> Optional[np.ndarray]:
        """Load node features."""
        if node_features is None:
            # Compute basic graph features
            features = np.zeros((self.num_nodes, 5))
            degrees = dict(self.graph.degree())
            clustering = nx.clustering(self.graph)

            for idx, node in enumerate(self.nodes):
                features[idx, 0] = degrees.get(node, 0)
                features[idx, 1] = clustering.get(node, 0)
                # Add more features as needed

            return features

        elif isinstance(node_features, str):
            # Load from file
            if node_features.endswith(".csv"):
                df = pd.read_csv(node_features)
                return df.values
            elif node_features.endswith(".npy"):
                return np.load(node_features)
            elif node_features.endswith(".pkl"):
                with open(node_features, "rb") as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported feature format: {node_features}")

        elif isinstance(node_features, pd.DataFrame):
            return node_features.values

        elif isinstance(node_features, np.ndarray):
            return node_features

        else:
            raise TypeError(f"Unsupported node_features type: {type(node_features)}")

    def _load_labels(
        self,
        labels: Optional[Union[str, pd.DataFrame, np.ndarray, Dict]]
    ) -> Optional[np.ndarray]:
        """Load node labels."""
        if labels is None:
            return None

        elif isinstance(labels, str):
            if labels.endswith(".csv"):
                df = pd.read_csv(labels)
                return df.values.flatten()
            elif labels.endswith(".npy"):
                return np.load(labels)
            else:
                raise ValueError(f"Unsupported label format: {labels}")

        elif isinstance(labels, pd.DataFrame):
            return labels.values.flatten()

        elif isinstance(labels, pd.Series):
            # pandas Series with node indices
            return labels.values

        elif isinstance(labels, np.ndarray):
            return labels

        elif isinstance(labels, dict):
            # Map node to label
            label_array = np.zeros(self.num_nodes)
            for node, label in labels.items():
                if node in self.node_to_idx:
                    label_array[self.node_to_idx[node]] = label
            return label_array

        else:
            raise TypeError(f"Unsupported labels type: {type(labels)}")

    def _compute_or_load_quantum_params(self) -> Dict:
        """
        Compute quantum encoding parameters for all nodes.

        Returns dict with:
        - squeezing: [num_nodes, max_nodes]
        - unitary: [num_nodes, max_nodes, max_nodes]
        - metadata: list of dicts
        """
        cache_path = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"quantum_params_m{self.max_nodes}_r{self.ego_radius}.pkl"

        # Try loading from cache
        if cache_path is not None and cache_path.exists():
            print(f"Loading cached quantum parameters from {cache_path}")
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if is_quantum_cache_usable(cached, max_nodes=self.max_nodes):
                return cached
            report = inspect_quantum_cache(cached, max_nodes=self.max_nodes)
            print(
                "Warning: Detected invalid/degenerate quantum cache. Recomputing...\n"
                f"  sq_zero_ratio={report.get('sq_zero_ratio', -1):.4f}, "
                f"unitary_identity_ratio={report.get('unitary_identity_ratio', -1):.4f}, "
                f"center_ratio={report.get('center_ratio', -1):.4f}"
            )

        # Compute quantum parameters
        from utils.graph_utils import (
            preprocess_graph_for_quantum,
            SubgraphConfig
        )

        print("Precomputing quantum parameters...")
        config = SubgraphConfig(
            max_nodes=self.max_nodes,
            radius=self.ego_radius,
            normalize=True
        )

        squeezing_list = []
        unitary_list = []
        metadata_list = []

        for node in self.nodes:
            try:
                squeeze, U, adj, meta = preprocess_graph_for_quantum(
                    self.graph, node, config
                )
                squeezing_list.append(squeeze)
                unitary_list.append(U)
                metadata_list.append(meta)

            except Exception as e:
                # Fallback for failed preprocessing
                print(f"Warning: Failed to process node {node}: {e}")
                squeezing_list.append(np.zeros(self.max_nodes))
                unitary_list.append(np.eye(self.max_nodes))
                metadata_list.append({"center_node": int(node), "error": str(e)})

        # Convert to arrays
        quantum_params = {
            "squeezing": np.array(squeezing_list, dtype=np.float32),
            "unitary": np.array(unitary_list, dtype=np.float32),
            "metadata": metadata_list
        }

        # Save to cache
        if cache_path is not None:
            print(f"Caching quantum parameters to {cache_path}")
            with open(cache_path, "wb") as f:
                pickle.dump(quantum_params, f)

        return quantum_params

    def __len__(self) -> int:
        return self.num_nodes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary with:
            - squeezing: [max_nodes]
            - unitary: [max_nodes, max_nodes]
            - classical_features: [feature_dim]
            - label: scalar (optional)
        """
        node = self.nodes[idx]

        # Get quantum parameters from cache by node id (not by local dataset index),
        # and compute on-demand when missing.
        cache_idx = self.cache_node_to_idx.get(int(node), None)
        if cache_idx is not None:
            squeezing_np = self.quantum_params["squeezing"][cache_idx]
            unitary_np = self.quantum_params["unitary"][cache_idx]
        else:
            cached = self._ondemand_quantum_cache.get(int(node))
            if cached is None:
                from utils.graph_utils import preprocess_graph_for_quantum, SubgraphConfig

                config = SubgraphConfig(max_nodes=self.max_nodes, radius=self.ego_radius, normalize=True)
                try:
                    squeeze, U, _, _ = preprocess_graph_for_quantum(self.graph, node, config)
                except Exception:
                    squeeze = np.zeros(self.max_nodes, dtype=np.float32)
                    U = np.eye(self.max_nodes, dtype=np.float32)
                cached = (squeeze.astype(np.float32), U.astype(np.float32))
                self._ondemand_quantum_cache[int(node)] = cached
            squeezing_np, unitary_np = cached

        squeezing = torch.from_numpy(squeezing_np)
        unitary = torch.from_numpy(unitary_np)

        # Get classical features
        if self.node_features is not None:
            classical = torch.from_numpy(self.node_features[idx]).float()
        else:
            classical = torch.zeros(1)

        # Get label
        label = None
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)

        sample = {
            "squeezing": squeezing,
            "unitary": unitary,
            "classical_features": classical,
            "node_id": node,
            "idx": idx
        }

        if label is not None:
            sample["label"] = label

        # Apply transform
        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class SyntheticFinancialDataset(FinancialGraphDataset):
    """
    Generate synthetic financial transaction graphs for testing.

    Creates graphs with planted fraud communities (dense subgraphs).
    """

    def __init__(
        self,
        num_nodes: int = 1000,
        fraud_ratio: float = 0.05,
        fraud_community_size: int = 20,
        fraud_density: float = 0.8,
        normal_density: float = 0.02,
        feature_dim: int = 10,
        **kwargs
    ):
        """
        Generate synthetic dataset.

        Args:
            num_nodes: Total number of nodes
            fraud_ratio: Fraction of fraudulent nodes
            fraud_community_size: Size of fraud communities
            fraud_density: Edge density within fraud communities
            normal_density: Edge density in normal network
            feature_dim: Dimension of node features
            **kwargs: Passed to parent class
        """
        # Generate graph
        G, labels = self._generate_synthetic_graph(
            num_nodes,
            fraud_ratio,
            fraud_community_size,
            fraud_density,
            normal_density
        )

        # Generate node features
        features = self._generate_synthetic_features(
            num_nodes,
            feature_dim,
            labels
        )

        # Initialize parent with edge list
        edge_list = list(G.edges(data=True))

        # Convert to DataFrame format expected by parent
        edge_df = pd.DataFrame(edge_list, columns=["source", "target", "data"])
        edge_df["weight"] = [d.get("weight", 1.0) for _, _, d in edge_list]
        edge_df = edge_df[["source", "target", "weight"]]

        # Initialize with pre-generated data
        self.graph = G
        self.labels = labels
        self.node_features = features
        self.nodes = list(G.nodes())
        self.num_nodes = len(self.nodes)
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        # Set other parameters
        self.max_nodes = kwargs.get("max_nodes", 20)
        self.ego_radius = kwargs.get("ego_radius", 1.5)
        self.cache_dir = None
        self.preload = True
        self.transform = None  # Transform function

        # Compute quantum parameters
        from utils.graph_utils import (
            preprocess_graph_for_quantum,
            SubgraphConfig
        )

        print("Precomputing quantum parameters for synthetic data...")
        config = SubgraphConfig(
            max_nodes=self.max_nodes,
            radius=self.ego_radius,
            normalize=True
        )

        squeezing_list = []
        unitary_list = []
        metadata_list = []

        for node in self.nodes:
            try:
                squeeze, U, adj, meta = preprocess_graph_for_quantum(
                    self.graph, node, config
                )
                squeezing_list.append(squeeze)
                unitary_list.append(U)
                metadata_list.append(meta)
            except Exception as e:
                squeezing_list.append(np.zeros(self.max_nodes))
                unitary_list.append(np.eye(self.max_nodes))
                metadata_list.append({"error": str(e)})

        self.quantum_params = {
            "squeezing": np.array(squeezing_list, dtype=np.float32),
            "unitary": np.array(unitary_list, dtype=np.float32),
            "metadata": metadata_list
        }

        print(f"Synthetic dataset created: {self.num_nodes} nodes, "
              f"{np.sum(labels)} fraud nodes")

    def _generate_synthetic_graph(
        self,
        num_nodes: int,
        fraud_ratio: float,
        fraud_community_size: int,
        fraud_density: float,
        normal_density: float
    ) -> Tuple[nx.Graph, np.ndarray]:
        """Generate synthetic graph with fraud communities."""
        G = nx.Graph()
        labels = np.zeros(num_nodes, dtype=int)

        # Add all nodes
        G.add_nodes_from(range(num_nodes))

        # Add normal edges (random graph)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.random.rand() < normal_density:
                    weight = np.random.rand()
                    G.add_edge(i, j, weight=weight)

        # Create fraud communities (dense subgraphs)
        num_fraud = int(num_nodes * fraud_ratio)
        num_communities = num_fraud // fraud_community_size

        fraud_nodes = np.random.choice(
            [n for n in range(num_nodes) if labels[n] == 0],
            num_fraud,
            replace=False
        )

        for c in range(num_communities):
            start = c * fraud_community_size
            end = start + fraud_community_size
            community = fraud_nodes[start:end]

            # Label as fraud
            for node in community:
                labels[node] = 1

            # Add dense internal edges
            for i in community:
                for j in community:
                    if i < j and np.random.rand() < fraud_density:
                        weight = np.random.rand() * 0.5 + 0.5  # Higher weights
                        G.add_edge(i, j, weight=weight)

        return G, labels

    def _generate_synthetic_features(
        self,
        num_nodes: int,
        feature_dim: int,
        labels: np.ndarray
    ) -> np.ndarray:
        """Generate synthetic node features."""
        features = np.random.randn(num_nodes, feature_dim).astype(np.float32)

        # Fraud nodes have different feature distribution
        fraud_indices = labels == 1
        features[fraud_indices] += np.random.randn(
            np.sum(fraud_indices),
            feature_dim
        ).astype(np.float32) * 0.5

        return features


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.

    Handles variable-sized classical features.
    """
    # Stack quantum parameters (fixed size)
    squeezing = torch.stack([item["squeezing"] for item in batch])
    unitary = torch.stack([item["unitary"] for item in batch])

    # Stack classical features (variable size)
    classical_feat_list = [item["classical_features"] for item in batch]

    # Pad to same size
    classical_features = torch.nn.utils.rnn.pad_sequence(
        classical_feat_list,
        batch_first=True
    )

    # Gather labels
    if "label" in batch[0]:
        labels = torch.stack([item["label"] for item in batch])
    else:
        labels = None

    result = {
        "squeezing": squeezing,
        "unitary": unitary,
        "classical_features": classical_features,
    }

    if labels is not None:
        result["label"] = labels

    return result


def load_elliptic_dataset(
    data_dir: str = "./data/elliptic",
    max_nodes: int = 20,
    ego_radius: float = 1.5,
    train_periods: Tuple[int, int] = (1, 34),
    test_periods: Tuple[int, int] = (35, 49),
    cache_dir: Optional[str] = None
) -> Tuple[FinancialGraphDataset, FinancialGraphDataset]:
    """
    Load Elliptic++ Bitcoin dataset for Q-GAD.

    This convenience function:
    1. Loads Elliptic++ dataset using EllipticPlusPlusDataset
    2. Splits by time period (temporal split)
    3. Creates FinancialGraphDataset instances for train/test

    Args:
        data_dir: Directory containing Elliptic++ dataset
        max_nodes: Maximum nodes for ego-net extraction
        ego_radius: Ego-network radius
        train_periods: Training time period range (1-49)
        test_periods: Testing time period range (1-49)
        cache_dir: Cache directory for preprocessed data

    Returns:
        (train_dataset, test_dataset) as FinancialGraphDataset instances

    Example:
        train_ds, test_ds = load_elliptic_dataset(
            data_dir="./data/elliptic",
            max_nodes=20
        )
    """
    from data.elliptic_dataset import EllipticPlusPlusDataset

    print("="*50)
    print("Loading Elliptic++ Dataset")
    print("="*50)

    # Load Elliptic++ dataset
    elliptic = EllipticPlusPlusDataset(data_dir=data_dir)

    if not elliptic.load_data():
        raise RuntimeError(
            f"Failed to load Elliptic++ dataset from {data_dir}. "
            f"Please ensure the following files exist:\n"
            f"  - {elliptic.edgelist_path}\n"
            f"  - {elliptic.features_path}\n"
            f"  - {elliptic.classes_path}\n"
            f"\nDownload from: https://www.kaggle.com/datasets/ellipticdata/transaction-data"
        )

    # Build graph
    elliptic.build_networkx_graph()

    # Print statistics
    elliptic.get_statistics()

    # Get temporal split
    train_nodes, test_nodes = elliptic.split_by_time(train_periods, test_periods)

    # Extract subgraphs for train/test
    train_graph = elliptic.graph.subgraph(train_nodes).copy()
    test_graph = elliptic.graph.subgraph(test_nodes).copy()

    print(f"\nCreating PyTorch datasets...")

    # Create train dataset
    train_dataset = FinancialGraphDataset(
        edge_list=train_graph,
        node_features=elliptic.features_df.loc[train_nodes],
        labels=elliptic.classes_df.loc[train_nodes, 'label'],
        max_nodes=max_nodes,
        ego_radius=ego_radius,
        cache_dir=cache_dir,
        preload=True
    )

    # Create test dataset
    test_dataset = FinancialGraphDataset(
        edge_list=test_graph,
        node_features=elliptic.features_df.loc[test_nodes],
        labels=elliptic.classes_df.loc[test_nodes, 'label'],
        max_nodes=max_nodes,
        ego_radius=ego_radius,
        cache_dir=cache_dir,
        preload=True
    )

    print(f"\nTrain dataset: {len(train_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")

    return train_dataset, test_dataset


if __name__ == "__main__":
    print("Testing Financial Graph Dataset...")

    # Create synthetic dataset
    dataset = SyntheticFinancialDataset(
        num_nodes=100,
        fraud_ratio=0.1,
        fraud_community_size=10,
        max_nodes=20,
        ego_radius=1.5
    )

    print(f"Dataset size: {len(dataset)}")

    # Test single sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Squeezing shape: {sample['squeezing'].shape}")
    print(f"Unitary shape: {sample['unitary'].shape}")
    print(f"Classical features shape: {sample['classical_features'].shape}")
    print(f"Label: {sample['label']}")

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    batch = next(iter(loader))
    print(f"\nBatch squeezing shape: {batch['squeezing'].shape}")
    print(f"Batch unitary shape: {batch['unitary'].shape}")
    print(f"Batch classical shape: {batch['classical_features'].shape}")
    print(f"Batch labels shape: {batch['label'].shape}")

    print("\nDataset test completed!")
