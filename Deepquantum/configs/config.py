"""
Configuration management for Q-GAD system.

Centralized configuration for all components.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    # Data paths
    edge_list_path: Optional[str] = None
    node_features_path: Optional[str] = None
    labels_path: Optional[str] = None
    cache_dir: str = "./data/cache"

    # Graph preprocessing
    max_nodes: int = 20  # Quantum modes
    ego_radius: float = 1.5  # Ego-network radius
    normalize_adj: bool = True

    # Synthetic data (for testing)
    use_synthetic: bool = False
    num_nodes: int = 1000
    fraud_ratio: float = 0.05
    fraud_community_size: int = 20
    fraud_density: float = 0.8
    normal_density: float = 0.02

    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # DataLoader
    batch_size: int = 16
    num_workers: int = 0  # Windows compatibility
    pin_memory: bool = False


@dataclass
class GBSConfig:
    """Gaussian Boson Sampling configuration."""
    n_modes: int = 20  # Number of quantum modes
    n_shots: int = 1000  # Samples per subgraph
    backend: str = "gaussian"  # 'gaussian' or 'fock'
    cutoff: int = 5  # Fock space cutoff

    # Noise model
    loss_rate: float = 0.0  # Photon loss (0 = no loss)
    thermal_noise: float = 0.0  # Thermal noise

    # Variational parameters
    use_displacement: bool = True
    use_kerr: bool = False
    max_squeezing: float = 2.0


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Quantum features
    quantum_feature_dim: int = 9

    # Classical features
    classical_feature_dim: int = 10

    # Neural network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.1

    # Fusion strategy
    use_xgboost_fusion: bool = False

    # Alternating training
    quantum_lr: float = 1e-3
    classifier_lr: float = 1e-3
    quantum_epochs_per_cycle: int = 1
    hybrid_epochs_per_cycle: int = 1


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training parameters
    n_epochs: int = 50
    early_stopping_patience: int = 10

    # Optimizer
    optimizer: str = "adam"
    weight_decay: float = 1e-5

    # Learning rate schedule
    use_lr_schedule: bool = False
    lr_decay_rate: float = 0.95
    lr_decay_steps: int = 5

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5  # Save every N epochs

    # Logging
    log_dir: str = "./logs"
    log_every: int = 1  # Log every N batches

    # Evaluation
    eval_every: int = 1  # Evaluate every N epochs

    # Device
    device: Optional[str] = None  # None = auto-detect

    # Alternating training (moved from ModelConfig)
    quantum_epochs_per_cycle: int = 1
    hybrid_epochs_per_cycle: int = 1


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    experiment_name: str = "qgad_experiment"
    seed: int = 42

    data: DataConfig = field(default_factory=DataConfig)
    gbs: GBSConfig = field(default_factory=GBSConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "data": self.data.__dict__,
            "gbs": self.gbs.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
        }

    def save(self, path: str):
        """Save configuration to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        print(f"Configuration saved to {path}")

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        # Recreate config from dict
        config = cls(
            experiment_name=data["experiment_name"],
            seed=data["seed"],
            data=DataConfig(**data["data"]),
            gbs=GBSConfig(**data["gbs"]),
            model=ModelConfig(**data["model"]),
            training=TrainingConfig(**data["training"])
        )

        print(f"Configuration loaded from {path}")
        return config


# Default configurations for different scenarios

def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()


def get_testing_config() -> ExperimentConfig:
    """Get configuration for quick testing."""
    config = ExperimentConfig()

    # Smaller data for testing
    config.data.use_synthetic = True
    config.data.num_nodes = 100
    config.data.batch_size = 8

    # Smaller model
    config.gbs.n_modes = 10
    config.gbs.n_shots = 100

    # Fewer epochs
    config.training.n_epochs = 5

    return config


def get_production_config() -> ExperimentConfig:
    """Get configuration for production training."""
    config = ExperimentConfig()

    # Larger data
    config.data.max_nodes = 30
    config.data.batch_size = 32

    # More samples
    config.gbs.n_modes = 30
    config.gbs.n_shots = 2000

    # Larger model
    config.model.hidden_dims = [128, 64, 32]

    # More training
    config.training.n_epochs = 100

    return config


if __name__ == "__main__":
    # Test configuration system
    print("Testing configuration system...")

    config = get_default_config()
    print(f"\nDefault config:")
    print(f"  Experiment: {config.experiment_name}")
    print(f"  Max nodes: {config.data.max_nodes}")
    print(f"  Quantum modes: {config.gbs.n_modes}")
    print(f"  Epochs: {config.training.n_epochs}")

    # Save and load
    config.save("configs/test_config.json")
    loaded = ExperimentConfig.load("configs/test_config.json")

    print(f"\nLoaded config matches: {loaded.experiment_name == config.experiment_name}")

    print("\nConfiguration test completed!")
