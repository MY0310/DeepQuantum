"""
Temporal Graph Dataset for Q-GAD system.

This module extends the base FinancialGraphDataset to support:
- Timestamped transaction graphs
- Temporal ego-network extraction
- Dynamic graph snapshots
- Time-aware train/test splits

For financial fraud detection, temporal information is crucial since:
- Fraud patterns evolve over time
- Transaction sequences reveal behavior
- Early detection requires temporal context
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from collections import defaultdict
import pickle

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.graph_utils import preprocess_graph_for_quantum, SubgraphConfig


class TemporalFinancialDataset(Dataset):
    """
    Temporal financial graph dataset with time-aware quantum encoding.

    Key features:
    - Time-stamped edges (transactions with timestamps)
    - Temporal ego-networks (considering time windows)
    - Dynamic snapshots (evolving graph over time)
    - Time-aware features (aggregated over time windows)
    """

    def __init__(
        self,
        edge_list: Union[str, pd.DataFrame, List[Tuple]],
        node_features: Optional[Union[str, pd.DataFrame, np.ndarray]] = None,
        labels: Optional[Union[str, pd.DataFrame, np.ndarray, Dict]] = None,
        max_nodes: int = 20,
        ego_radius: float = 1.5,
        time_window: Optional[float] = None,
        num_snapshots: Optional[int] = None,
        cache_dir: Optional[str] = None,
        preload: bool = True
    ):
        """
        Initialize temporal dataset.

        Args:
            edge_list: Path to CSV/DataFrame with columns [source, target, timestamp, weight]
                       or list of (source, target, timestamp, weight) tuples
            node_features: Path to CSV/DataFrame or array of node features
            labels: Path to CSV/DataFrame or dict/array of node labels
            max_nodes: Maximum nodes per subgraph
            ego_radius: Ego-network extraction radius
            time_window: Time window for temporal ego-net (e.g., 7 days)
            num_snapshots: Number of temporal snapshots (if using snapshot approach)
            cache_dir: Directory to cache preprocessed data
            preload: Whether to preload all data
        """
        super().__init__()
        self.max_nodes = max_nodes
        self.ego_radius = ego_radius
        self.time_window = time_window
        self.num_snapshots = num_snapshots
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.preload = preload

        # Load temporal graph
        self.temporal_graph = self._load_temporal_graph(edge_list)

        # Load node features
        self.node_features = self._load_node_features(node_features)

        # Load labels
        self.labels = self._load_labels(labels)

        # Get node list
        self.nodes = list(self.temporal_graph.nodes())
        self.num_nodes = len(self.nodes)

        # Create node to index mapping
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}

        # Get time range
        self._compute_time_statistics()

        # Precompute quantum parameters with temporal context
        self.quantum_params = self._compute_or_load_quantum_params()

        print(f"Temporal dataset initialized: {self.num_nodes} nodes, "
              f"{self.temporal_graph.number_of_edges()} edges")
        print(f"  Time range: {self.min_time:.2f} to {self.max_time:.2f}")
        if self.time_window:
            print(f"  Time window: {self.time_window}")

    def _load_temporal_graph(self, edge_list: Union[str, pd.DataFrame, List]) -> nx.DiGraph:
        """Load temporal graph with timestamped edges."""
        if isinstance(edge_list, str):
            # Load from file
            if edge_list.endswith(".csv"):
                df = pd.read_csv(edge_list)
                required_cols = ['txId1', 'txId2']  # Elliptic format
                time_col = 'time' if 'time' in df.columns else 'timestamp'

                if all(col in df.columns for col in required_cols):
                    # Elliptic format
                    G = nx.from_pandas_edgelist(
                        df,
                        source='txId1',
                        target='txId2',
                        edge_attr=[time_col] + (['weight'] if 'weight' in df.columns else []),
                        create_using=nx.DiGraph()
                    )
                else:
                    # Generic format
                    cols = df.columns.tolist()
                    G = nx.from_pandas_edgelist(
                        df,
                        source=cols[0],
                        target=cols[1],
                        edge_attr=cols[2:] if len(cols) > 2 else None,
                        create_using=nx.DiGraph()
                    )
            else:
                raise ValueError(f"Unsupported file format: {edge_list}")
        elif isinstance(edge_list, pd.DataFrame):
            G = nx.from_pandas_edgelist(
                edge_list,
                create_using=nx.DiGraph()
            )
        elif isinstance(edge_list, list):
            # List of (source, target, timestamp, weight)
            G = nx.DiGraph()
            for edge in edge_list:
                if len(edge) >= 3:
                    source, target, timestamp = edge[:3]
                    weight = edge[3] if len(edge) > 3 else 1.0
                    G.add_edge(source, target, timestamp=timestamp, weight=weight)
        else:
            raise ValueError(f"Unsupported edge_list type: {type(edge_list)}")

        # Ensure timestamp attribute exists
        for u, v, data in G.edges(data=True):
            if 'timestamp' not in data and 'time' in data:
                data['timestamp'] = data['time']
            elif 'timestamp' not in data:
                data['timestamp'] = 0

        return G

    def _load_node_features(self, node_features: Union[str, pd.DataFrame, np.ndarray, None]) -> np.ndarray:
        """Load node features."""
        if node_features is None:
            # Generate basic structural features
            return self._generate_structural_features()
        elif isinstance(node_features, str):
            df = pd.read_csv(node_features, index_col=0)
            return df.values
        elif isinstance(node_features, pd.DataFrame):
            return node_features.values
        elif isinstance(node_features, np.ndarray):
            return node_features
        else:
            raise ValueError(f"Unsupported node_features type: {type(node_features)}")

    def _generate_structural_features(self) -> np.ndarray:
        """Generate basic structural features for each node."""
        features = []
        for node in self.nodes:
            # Basic degree features
            in_degree = self.temporal_graph.in_degree(node)
            out_degree = self.temporal_graph.out_degree(node)
            total_degree = in_degree + out_degree

            # Temporal features (if available)
            timestamps = []
            for _, _, data in self.temporal_graph.edges(node, data=True):
                if 'timestamp' in data:
                    timestamps.append(data['timestamp'])

            time_mean = np.mean(timestamps) if timestamps else 0
            time_std = np.std(timestamps) if timestamps else 0

            features.append([in_degree, out_degree, total_degree, time_mean, time_std])

        return np.array(features, dtype=np.float32)

    def _load_labels(self, labels: Union[str, pd.DataFrame, np.ndarray, Dict, None]) -> np.ndarray:
        """Load node labels."""
        if labels is None:
            return np.zeros(self.num_nodes, dtype=int)
        elif isinstance(labels, dict):
            return np.array([labels.get(node, 0) for node in self.nodes], dtype=int)
        elif isinstance(labels, (str, pd.DataFrame)):
            if isinstance(labels, str):
                df = pd.read_csv(labels, index_col=0)
            else:
                df = labels
            return np.array([df.loc[node, 'label'] if node in df.index else 0
                           for node in self.nodes], dtype=int)
        elif isinstance(labels, np.ndarray):
            return labels
        else:
            raise ValueError(f"Unsupported labels type: {type(labels)}")

    def _compute_time_statistics(self):
        """Compute time range statistics."""
        timestamps = []
        for _, _, data in self.temporal_graph.edges(data=True):
            if 'timestamp' in data:
                timestamps.append(data['timestamp'])

        if timestamps:
            self.min_time = min(timestamps)
            self.max_time = max(timestamps)
            self.mean_time = np.mean(timestamps)
            self.std_time = np.std(timestamps)
        else:
            self.min_time = self.max_time = self.mean_time = self.std_time = 0

    def _compute_or_load_quantum_params(self) -> Dict:
        """Compute or load quantum parameters with temporal context."""
        cache_path = None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / "temporal_quantum_params.pkl"

            if cache_path.exists():
                print("Loading cached quantum parameters...")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

        print("Computing quantum parameters with temporal context...")
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
                # Get temporal context for this node
                if self.time_window:
                    # Extract temporal subgraph
                    subgraph = self._extract_temporal_ego_net(node, self.time_window)
                else:
                    # Use full graph
                    subgraph = self.temporal_graph

                # Preprocess for quantum encoding
                squeeze, U, adj, meta = preprocess_graph_for_quantum(
                    subgraph, node, config
                )
                squeezing_list.append(squeeze)
                unitary_list.append(U)

                # Add temporal metadata
                meta['temporal_stats'] = self._get_node_temporal_stats(node)
                metadata_list.append(meta)

            except Exception as e:
                squeezing_list.append(np.zeros(self.max_nodes))
                unitary_list.append(np.eye(self.max_nodes))
                metadata_list.append({"error": str(e)})

        result = {
            "squeezing": np.array(squeezing_list, dtype=np.float32),
            "unitary": np.array(unitary_list, dtype=np.float32),
            "metadata": metadata_list
        }

        # Cache results
        if cache_path:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)

        return result

    def _extract_temporal_ego_net(self, node: str, time_window: float) -> nx.DiGraph:
        """Extract temporal ego-network considering time window."""
        # Get ego network
        ego_net = nx.ego_graph(self.temporal_graph, node, radius=int(self.ego_radius))

        # Get node's timestamp
        node_time = self._get_node_timestamp(node)

        # Filter edges by time window
        edges_to_keep = []
        for u, v, data in ego_net.edges(data=True):
            edge_time = data.get('timestamp', 0)
            if abs(edge_time - node_time) <= time_window:
                edges_to_keep.append((u, v))

        # Create filtered subgraph
        temporal_subgraph = ego_net.edge_subgraph(edges_to_keep).copy()

        return temporal_subgraph

    def _get_node_timestamp(self, node: str) -> float:
        """Get representative timestamp for a node."""
        timestamps = []
        for _, _, data in self.temporal_graph.edges(node, data=True):
            if 'timestamp' in data:
                timestamps.append(data['timestamp'])

        return np.mean(timestamps) if timestamps else 0

    def _get_node_temporal_stats(self, node: str) -> Dict:
        """Get temporal statistics for a node."""
        timestamps = []
        weights = []

        for _, _, data in self.temporal_graph.edges(node, data=True):
            if 'timestamp' in data:
                timestamps.append(data['timestamp'])
            if 'weight' in data:
                weights.append(data['weight'])

        stats = {
            'timestamp_mean': np.mean(timestamps) if timestamps else 0,
            'timestamp_std': np.std(timestamps) if timestamps else 0,
            'timestamp_min': min(timestamps) if timestamps else 0,
            'timestamp_max': max(timestamps) if timestamps else 0,
            'weight_mean': np.mean(weights) if weights else 0,
            'num_transactions': len(timestamps)
        }

        return stats

    def __len__(self) -> int:
        return self.num_nodes

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        node = self.nodes[idx]

        # Quantum parameters
        squeezing = torch.tensor(self.quantum_params["squeezing"][idx], dtype=torch.float32)
        unitary = torch.tensor(self.quantum_params["unitary"][idx], dtype=torch.float32)

        # Classical features
        if self.node_features is not None:
            classical = torch.tensor(self.node_features[idx], dtype=torch.float32)
        else:
            classical = torch.zeros(10, dtype=torch.float32)

        # Label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        # Temporal metadata
        metadata = self.quantum_params["metadata"][idx]

        return {
            "squeezing": squeezing,
            "unitary": unitary,
            "classical_features": classical,
            "label": label,
            "node_id": node,
            "temporal_stats": metadata.get('temporal_stats', {})
        }

    def get_temporal_snapshots(self, num_snapshots: int) -> List['TemporalFinancialDataset']:
        """
        Split dataset into temporal snapshots.

        Args:
            num_snapshots: Number of time snapshots

        Returns:
            List of TemporalFinancialDataset, one per snapshot
        """
        if not hasattr(self, 'min_time') or self.min_time == self.max_time:
            raise ValueError("No temporal information available")

        # Compute time boundaries
        time_range = self.max_time - self.min_time
        snapshot_duration = time_range / num_snapshots

        snapshots = []
        for i in range(num_snapshots):
            start_time = self.min_time + i * snapshot_duration
            end_time = start_time + snapshot_duration

            # Filter nodes by time
            snapshot_nodes = []
            for node in self.nodes:
                node_time = self._get_node_timestamp(node)
                if start_time <= node_time < end_time:
                    snapshot_nodes.append(node)

            # Create snapshot dataset
            snapshot = TemporalFinancialDataset(
                edge_list=self.temporal_graph.subgraph(snapshot_nodes),
                node_features=self.node_features,
                labels={node: self.labels[self.node_to_idx[node]] for node in snapshot_nodes},
                max_nodes=self.max_nodes,
                ego_radius=self.ego_radius,
                cache_dir=None,
                preload=False
            )

            snapshots.append(snapshot)

        return snapshots


def create_temporal_dataset_from_elliptic(
    data_dir: str = "./data/elliptic",
    max_nodes: int = 20,
    ego_radius: float = 1.5,
    time_window: Optional[float] = None,
    use_snapshots: bool = False,
    num_snapshots: int = 5
) -> Union[TemporalFinancialDataset, List[TemporalFinancialDataset]]:
    """
    Create temporal dataset from Elliptic++ Bitcoin data.

    Args:
        data_dir: Directory containing Elliptic++ dataset
        max_nodes: Maximum nodes for ego-net extraction
        ego_radius: Ego-network radius
        time_window: Time window for temporal ego-net (in time steps)
        use_snapshots: Whether to split into temporal snapshots
        num_snapshots: Number of snapshots if use_snapshots=True

    Returns:
        TemporalFinancialDataset or list of snapshots
    """
    from data.elliptic_dataset import EllipticPlusPlusDataset

    # Load Elliptic++ dataset
    elliptic = EllipticPlusPlusDataset(data_dir=data_dir)
    if not elliptic.load_data():
        raise RuntimeError(f"Failed to load Elliptic++ dataset from {data_dir}")

    elliptic.build_networkx_graph()

    # Create temporal dataset
    dataset = TemporalFinancialDataset(
        edge_list=elliptic.edgelist_path,
        node_features=elliptic.features_path,
        labels=elliptic.classes_path,
        max_nodes=max_nodes,
        ego_radius=ego_radius,
        time_window=time_window,
        cache_dir=f"{data_dir}/processed/temporal",
        preload=True
    )

    if use_snapshots:
        return dataset.get_temporal_snapshots(num_snapshots)
    else:
        return dataset


if __name__ == "__main__":
    print("Testing Temporal Financial Dataset...")

    # Test with synthetic temporal data
    print("Creating synthetic temporal graph...")
    G = nx.DiGraph()

    # Add nodes and timestamped edges
    for i in range(100):
        G.add_node(f"tx_{i}")

    # Add timestamped transactions
    import random
    for i in range(500):
        source = f"tx_{random.randint(0, 99)}"
        target = f"tx_{random.randint(0, 99)}"
        timestamp = random.uniform(0, 49)  # 49 time steps like Elliptic
        weight = random.random()
        G.add_edge(source, target, timestamp=timestamp, weight=weight)

    # Create labels
    labels = {f"tx_{i}": 1 if random.random() < 0.1 else 0 for i in range(100)}

    # Create temporal dataset
    dataset = TemporalFinancialDataset(
        edge_list=G,
        labels=labels,
        max_nodes=20,
        ego_radius=1.5,
        time_window=5.0
    )

    print(f"Dataset size: {len(dataset)}")

    # Test sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Squeezing shape: {sample['squeezing'].shape}")
    print(f"Temporal stats: {sample['temporal_stats']}")

    print("\nTemporal dataset test completed!")
