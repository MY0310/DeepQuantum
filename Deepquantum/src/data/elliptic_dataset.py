"""
Elliptic++ Bitcoin dataset downloader and processor.

This module provides utilities to:
- Download Elliptic++ dataset from Kaggle
- Parse and process the Bitcoin transaction graph
- Extract features and labels for fraud detection

Dataset Info:
- 203,769 nodes (Bitcoin transactions)
- 234,355 edges (transaction flows)
- 2 classes: illicit (fraud) vs licit (normal)
- Time periods: 49 time steps (1 week each)
- Features: 166 features per node (aggregated transaction info)
"""

import os
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# Optional imports for downloading
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    requests = None

try:
    import zipfile
    HAS_ZIPFILE = True
except ImportError:
    HAS_ZIPFILE = False
    zipfile = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.graph_utils import preprocess_graph_for_quantum, SubgraphConfig


class EllipticPlusPlusDataset:
    """
    Elliptic++ Bitcoin dataset for financial fraud detection.

    The dataset consists of:
    - edgelist.csv: Transaction graph (source, target, timestamp)
    - features.csv: Node features (166 aggregated features)
    - classes.csv: Node labels (0=unknown, 1=illicit, 2=licit)

    Paper: "Elliptic++: An Enhanced Bitcoin Transaction Network Dataset
            for Anti-Money Laundering and Financial Fraud Detection"
    """

    KAGGLE_DATASET = "ellipticdata/transaction-data"

    def __init__(
        self,
        data_dir: str = "./data/elliptic",
        force_download: bool = False,
        cache_processed: bool = True
    ):
        """
        Initialize Elliptic++ dataset handler.

        Args:
            data_dir: Directory to store/lookup dataset
            force_download: Force re-download even if files exist
            cache_processed: Cache processed graph data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.cache_processed = cache_processed
        self.force_download = force_download

        # File paths
        self.edgelist_path = self.raw_dir / "edgelist.csv"
        self.features_path = self.raw_dir / "features.csv"
        self.classes_path = self.raw_dir / "classes.csv"

        # Load data
        self.edges_df = None
        self.features_df = None
        self.classes_df = None
        self.graph = None

    def download_from_kaggle(self, api_key: Optional[str] = None) -> bool:
        """
        Download Elliptic++ dataset from Kaggle.

        Args:
            api_key: Kaggle API key (optional, can use ~/.kaggle/kaggle.json)

        Returns:
            True if download successful, False otherwise
        """
        try:
            import kaggle
        except ImportError:
            print("Kaggle API not installed. Install with: pip install kaggle")
            print("Or download manually from: https://www.kaggle.com/datasets/ellipticdata/transaction-data")
            return False

        print("Downloading Elliptic++ dataset from Kaggle...")

        try:
            # Download dataset
            kaggle.api.dataset_download_files(
                self.KAGGLE_DATASET,
                path=str(self.raw_dir),
                unzip=True,
                force=self.force_download
            )
            print(f"Dataset downloaded to {self.raw_dir}")
            return True
        except Exception as e:
            print(f"Failed to download from Kaggle: {e}")
            print("\nAlternative: Manual download")
            print("1. Visit: https://www.kaggle.com/datasets/ellipticdata/transaction-data")
            print(f"2. Extract to: {self.raw_dir}")
            return False

    def download_from_mirror(self, mirror_url: Optional[str] = None) -> bool:
        """
        Download from alternative mirror (if available).

        Args:
            mirror_url: URL to download from

        Returns:
            True if successful
        """
        if mirror_url is None:
            print("No mirror URL provided. Using official Kaggle source.")
            return self.download_from_kaggle()

        print(f"Downloading from mirror: {mirror_url}")
        try:
            response = requests.get(mirror_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024

            zip_path = self.raw_dir / "elliptic_dataset.zip"
            with open(zip_path, 'wb') as f:
                for data in tqdm(response.iter_content(block_size),
                                 total=total_size // block_size,
                                 unit='KB'):
                    f.write(data)

            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.raw_dir)

            os.remove(zip_path)
            print("Dataset downloaded and extracted successfully")
            return True

        except Exception as e:
            print(f"Failed to download from mirror: {e}")
            return False

    def load_data(self) -> bool:
        """
        Load the Elliptic++ dataset from disk.

        Returns:
            True if data loaded successfully
        """
        # Check if files exist
        required_files = [self.edgelist_path, self.features_path, self.classes_path]
        if not all(f.exists() for f in required_files):
            print("Dataset files not found. Please download first:")
            print("  dataset.download_from_kaggle() or")
            print("  dataset.download_from_mirror(url)")
            return False

        print("Loading Elliptic++ dataset...")

        # Load edge list
        print("  - Loading edge list...")
        self.edges_df = pd.read_csv(self.edgelist_path)
        print(f"    {len(self.edges_df)} edges loaded")
        print(f"    Columns: {self.edges_df.columns.tolist()}")

        # Load features (first column is txId, need to set as index)
        print("  - Loading features...")
        self.features_df = pd.read_csv(self.features_path, index_col=0)
        print(f"    {len(self.features_df)} nodes, {self.features_df.shape[1]} features")

        # Load classes
        print("  - Loading classes...")
        self.classes_df = pd.read_csv(self.classes_path, index_col=0)
        print(f"    {len(self.classes_df)} labeled nodes")
        print(f"    Class values: {self.classes_df['class'].unique()}")

        # Filter out unknown nodes
        print("  - Filtering unknown nodes...")
        # Handle both string "unknown" and numeric 0
        if self.classes_df['class'].dtype == object:
            labeled_mask = self.classes_df['class'] != 'unknown'
            # Convert string labels to numeric without pandas downcast warning.
            self.classes_df['class'] = (
                self.classes_df['class']
                .map({'1': 1, '2': 2, 'unknown': 0})
                .pipe(pd.to_numeric, errors='coerce')
                .fillna(0)
                .astype(np.int64)
            )
        else:
            labeled_mask = self.classes_df['class'] != 0

        labeled_nodes = self.classes_df[labeled_mask].index

        # Ensure features only include labeled nodes that exist
        common_nodes = labeled_nodes.intersection(self.features_df.index)
        self.features_df = self.features_df.loc[common_nodes]
        self.classes_df = self.classes_df.loc[common_nodes]

        # Convert labels to binary fraud target:
        # 1 -> 1 (illicit/fraud), 2 -> 0 (licit/normal)
        self.classes_df['label'] = self.classes_df['class'].map({1: 1, 2: 0})

        print(f"    Final: {len(self.features_df)} labeled nodes")
        print(f"      Licit (normal): {(self.classes_df['label'] == 0).sum()}")
        print(f"      Illicit (fraud): {(self.classes_df['label'] == 1).sum()}")

        return True

    def build_networkx_graph(self, min_period: int = 1, max_period: int = 49):
        """
        Build NetworkX graph from edge list (filtered by time period).

        Args:
            min_period: Minimum time period (1-49)
            max_period: Maximum time period (1-49)
        """
        import networkx as nx

        print(f"Building NetworkX graph (periods {min_period}-{max_period})...")

        # Filter edges by time period
        if 'time' in self.edges_df.columns:
            edges_filtered = self.edges_df[
                (self.edges_df['time'] >= min_period) &
                (self.edges_df['time'] <= max_period)
            ]
        else:
            edges_filtered = self.edges_df

        # Create directed graph
        self.graph = nx.from_pandas_edgelist(
            edges_filtered,
            source='txId1',
            target='txId2',
            edge_attr='weight' if 'weight' in edges_filtered.columns else None,
            create_using=nx.DiGraph()
        )

        # Filter to only labeled nodes
        labeled_nodes = set(self.features_df.index)
        self.graph = self.graph.subgraph(labeled_nodes).copy()

        print(f"  Graph created: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

        return self.graph

    def get_node_labels(self) -> pd.Series:
        """Get node labels as pandas Series."""
        return self.classes_df['label']

    def get_node_features(self) -> pd.DataFrame:
        """Get node features as DataFrame."""
        return self.features_df

    def split_by_time(self,
                     train_periods: Tuple[int, int] = (1, 34),
                     test_periods: Tuple[int, int] = (35, 49)
                     ) -> Tuple[pd.Index, pd.Index]:
        """
        Split nodes by time period (temporal split).

        Args:
            train_periods: (start, end) for training
            test_periods: (start, end) for testing

        Returns:
            (train_nodes, test_nodes) as pandas Index
        """
        # Try to get time information from edge list
        if 'time' in self.edges_df.columns:
            # Method 1: Time is in edge list
            train_edges = self.edges_df[
                (self.edges_df['time'] >= train_periods[0]) &
                (self.edges_df['time'] <= train_periods[1])
            ]
            test_edges = self.edges_df[
                (self.edges_df['time'] >= test_periods[0]) &
                (self.edges_df['time'] <= test_periods[1])
            ]

            train_nodes = set(train_edges['txId1']) | set(train_edges['txId2'])
            test_nodes = set(test_edges['txId1']) | set(test_edges['txId2'])
        else:
            # Method 2: Time is in features (column 1, second column)
            # Features format: index=txId, column 0 = time step, columns 1-165 = features
            if self.features_df is not None and len(self.features_df.columns) > 1:
                # Get time column (column name might be the time value or index 1)
                time_col = self.features_df.columns[0]  # First column after index is time

                train_mask = (self.features_df[time_col] >= train_periods[0]) & \
                            (self.features_df[time_col] <= train_periods[1])
                test_mask = (self.features_df[time_col] >= test_periods[0]) & \
                           (self.features_df[time_col] <= test_periods[1])

                train_nodes = set(self.features_df[train_mask].index)
                test_nodes = set(self.features_df[test_mask].index)

                print(f"Using time information from features (column: {time_col})")
            else:
                # Method 3: Random split if no time information available
                print("Warning: No time information available, using random split")
                import random
                all_nodes = list(self.features_df.index)
                random.seed(42)
                random.shuffle(all_nodes)

                split_point = int(0.7 * len(all_nodes))
                train_nodes = set(all_nodes[:split_point])
                test_nodes = set(all_nodes[split_point:])

                print(f"Random split (no temporal information)")

        # Filter to labeled nodes
        labeled = set(self.features_df.index)
        train_nodes = pd.Index(list(train_nodes & labeled))
        test_nodes = pd.Index(list(test_nodes & labeled))

        print(f"Time-based split:")
        print(f"  Train (periods {train_periods[0]}-{train_periods[1]}): {len(train_nodes)} nodes")
        print(f"  Test (periods {test_periods[0]}-{test_periods[1]}): {len(test_nodes)} nodes")

        return train_nodes, test_nodes

    def export_for_qgad(self,
                       output_dir: str,
                       max_nodes: int = 20,
                       ego_radius: float = 1.5
                       ) -> Path:
        """
        Export dataset in Q-GAD format (edge list + features + labels).

        Args:
            output_dir: Output directory
            max_nodes: Max nodes for ego-net extraction
            ego_radius: Ego-net radius

        Returns:
            Path to processed data
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export edge list
        edge_out = output_path / "edges.csv"
        if not self.edgelist_path.exists():
            self.edges_df.to_csv(edge_out, index=False)
        else:
            import shutil
            shutil.copy(self.edgelist_path, edge_out)

        # Export features
        feature_out = output_path / "features.csv"
        self.features_df.to_csv(feature_out)

        # Export labels
        label_out = output_path / "labels.csv"
        self.classes_df[['label']].to_csv(label_out)

        print(f"\nDataset exported for Q-GAD: {output_path}")
        print("  - edges.csv")
        print("  - features.csv")
        print("  - labels.csv")

        return output_path

    def get_statistics(self):
        """Print dataset statistics."""
        if self.classes_df is None:
            print("Dataset not loaded. Call load_data() first.")
            return

        print("\n" + "="*50)
        print("Elliptic++ Dataset Statistics")
        print("="*50)

        # Class distribution
        n_licit = (self.classes_df['label'] == 0).sum()
        n_illicit = (self.classes_df['label'] == 1).sum()
        n_total = len(self.classes_df)

        print(f"\nClass Distribution:")
        print(f"  Licit (normal):   {n_licit:6d} ({n_licit/n_total*100:.1f}%)")
        print(f"  Illicit (fraud):  {n_illicit:6d} ({n_illicit/n_total*100:.1f}%)")
        print(f"  Total:            {n_total:6d}")

        # Graph info
        if self.graph is not None:
            print(f"\nGraph Information:")
            print(f"  Nodes: {self.graph.number_of_nodes()}")
            print(f"  Edges: {self.graph.number_of_edges()}")
            print(f"  Avg degree: {sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes():.2f}")

            # Connected components
            import networkx as nx
            n_components = nx.number_weakly_connected_components(self.graph)
            print(f"  Weakly connected components: {n_components}")

        # Feature info
        print(f"\nFeature Information:")
        print(f"  Dimensions: {self.features_df.shape[1]}")
        print(f"  Range: [{self.features_df.min().min():.2f}, {self.features_df.max().max():.2f}]")

        print("="*50 + "\n")


def download_elliptic_dataset(data_dir: str = "./data/elliptic",
                               force: bool = False) -> EllipticPlusPlusDataset:
    """
    Convenience function to download and load Elliptic++ dataset.

    Args:
        data_dir: Directory to store dataset
        force: Force re-download

    Returns:
        EllipticPlusPlusDataset instance
    """
    dataset = EllipticPlusPlusDataset(data_dir=data_dir, force_download=force)

    # Try to download if files don't exist
    if not all(f.exists() for f in [
        dataset.edgelist_path,
        dataset.features_path,
        dataset.classes_path
    ]):
        print("Dataset files not found. Attempting download...")
        if not dataset.download_from_kaggle():
            print("\nPlease download manually:")
            print("1. Visit: https://www.kaggle.com/datasets/ellipticdata/transaction-data")
            print(f"2. Extract files to: {dataset.raw_dir}")
            print("3. Run: dataset.load_data()")
            return dataset

    # Load data
    dataset.load_data()

    return dataset


if __name__ == "__main__":
    import networkx as nx

    print("Testing Elliptic++ dataset handler...")

    # Create dataset instance
    dataset = EllipticPlusPlusDataset(data_dir="./data/elliptic_test")

    # Check if files exist
    if dataset.edgelist_path.exists():
        print("Found existing dataset files")
        dataset.load_data()
        dataset.build_networkx_graph()
        dataset.get_statistics()
    else:
        print("Dataset files not found.")
        print("To download, use:")
        print("  python elliptic_dataset.py --download")
        print("\nOr manually from Kaggle:")
        print("  https://www.kaggle.com/datasets/ellipticdata/transaction-data")
