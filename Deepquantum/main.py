"""
Main entry point for Q-GAD system.

This script provides:
- Training pipeline
- Evaluation pipeline
- Inference/prediction pipeline
"""

import torch
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from configs.config import ExperimentConfig, get_testing_config
from data.financial_dataset import (
    FinancialGraphDataset,
    SyntheticFinancialDataset,
    collate_fn
)
from trainer import QGADTrainer, create_model_and_trainer
from utils.helpers import (
    set_seed, get_device, print_metrics,
    plot_training_history, plot_confusion_matrix
)
from torch.utils.data import random_split, DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Q-GAD: Quantum Graph Anomaly Detector")

    # Mode
    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "evaluate", "predict"],
        help="Running mode"
    )

    # Configuration
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to configuration JSON file"
    )
    parser.add_argument(
        "--preset", type=str, default="testing",
        choices=["testing", "default", "production"],
        help="Configuration preset"
    )

    # Data
    parser.add_argument(
        "--data_dir", type=str, default="./data",
        help="Data directory"
    )
    parser.add_argument(
        "--use_synthetic", action="store_true",
        help="Use synthetic data for testing"
    )
    parser.add_argument(
        "--synthetic_nodes", type=int, default=500,
        help="Number of nodes for synthetic data"
    )

    # Training
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (cuda/cpu/mps)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    # Model
    parser.add_argument(
        "--n_modes", type=int, default=None,
        help="Number of quantum modes"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint for evaluation/resuming"
    )

    # Output
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--experiment_name", type=str, default="qgad_exp",
        help="Experiment name"
    )

    # XGBoost fusion
    parser.add_argument(
        "--use_xgboost", action="store_true",
        help="Use XGBoost fusion after neural training"
    )

    # GPU parallel training
    parser.add_argument(
        "--use_parallel", action="store_true",
        help="Enable multi-GPU parallel training"
    )
    parser.add_argument(
        "--use_ddp", action="store_true",
        help="Use DistributedDataParallel instead of DataParallel"
    )

    return parser.parse_args()


def load_dataset(args, config):
    """Load dataset based on arguments and configuration."""
    print("\n" + "="*50)
    print("Loading Dataset")
    print("="*50)

    use_synthetic = args.use_synthetic or config.data.use_synthetic

    if use_synthetic:
        print("Using synthetic dataset...")
        num_nodes = args.synthetic_nodes if args.synthetic_nodes > 0 else config.data.num_nodes

        dataset = SyntheticFinancialDataset(
            num_nodes=num_nodes,
            fraud_ratio=config.data.fraud_ratio,
            fraud_community_size=config.data.fraud_community_size,
            fraud_density=config.data.fraud_density,
            normal_density=config.data.normal_density,
            max_nodes=config.data.max_nodes,
            ego_radius=config.data.ego_radius
        )

        print(f"Synthetic dataset created: {len(dataset)} nodes")

    else:
        print("Loading real dataset...")
        edge_list_path = args.data_dir + "/edges.csv" if args.data_dir else config.data.edge_list_path
        features_path = args.data_dir + "/features.csv" if args.data_dir else config.data.node_features_path
        labels_path = args.data_dir + "/labels.csv" if args.data_dir else config.data.labels_path

        if edge_list_path is None:
            raise ValueError("Must specify edge_list_path in config or via --data_dir")

        dataset = FinancialGraphDataset(
            edge_list=edge_list_path,
            node_features=features_path,
            labels=labels_path,
            max_nodes=config.data.max_nodes,
            ego_radius=config.data.ego_radius,
            cache_dir=config.data.cache_dir
        )

        print(f"Dataset loaded: {len(dataset)} nodes")

    return dataset


def setup_data_loaders(dataset, config):
    """Split dataset and create data loaders."""
    print("\n" + "="*50)
    print("Setting Up Data Loaders")
    print("="*50)

    # Split dataset
    total_size = len(dataset)
    train_size = int(config.data.train_ratio * total_size)
    val_size = int(config.data.val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.data.num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.data.num_workers
    )

    return train_loader, val_loader, test_loader


def train(args, config):
    """Run training pipeline."""
    print("\n" + "="*50)
    print("Q-GAD Training Pipeline")
    print("="*50)

    # Set seed
    set_seed(args.seed if args.seed else config.seed)

    # Get device
    device = get_device(args.device if args.device else config.training.device)
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset(args, config)
    train_loader, val_loader, test_loader = setup_data_loaders(dataset, config)

    # Create model and trainer
    print("\n" + "="*50)
    print("Creating Model")
    print("="*50)

    n_modes = args.n_modes if args.n_modes else config.gbs.n_modes
    classical_dim = dataset.node_features.shape[1] if dataset.node_features is not None else 10

    model, trainer = create_model_and_trainer(
        n_modes=n_modes,
        classical_feature_dim=classical_dim,
        device=device,
        use_xgboost=args.use_xgboost,
        use_parallel=args.use_parallel,
        use_ddp=args.use_ddp
    )

    # Print model info
    from src.utils.helpers import count_parameters
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")

    # Load checkpoint if specified
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)

    # Override config from args
    n_epochs = args.epochs if args.epochs else config.training.n_epochs

    # Train
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)

    # Check if using mock implementation (no real quantum backend)
    if not model.quantum_extractor.gbs_kernel.has_deepquantum:
        print("Note: Using mock quantum implementation - skipping quantum kernel training")
        print("      Only training hybrid classifier...")
        quantum_epochs = 0
        hybrid_epochs = n_epochs
    else:
        quantum_epochs = config.training.quantum_epochs_per_cycle
        hybrid_epochs = config.training.hybrid_epochs_per_cycle

    if args.use_xgboost:
        # Train with XGBoost fusion
        xgb_model = trainer.train_with_xgboost_fusion(
            train_loader, val_loader,
            neural_epochs=n_epochs
        )
    else:
        # Standard alternating training
        trainer.train_alternating(
            train_loader, val_loader,
            n_epochs=n_epochs,
            quantum_epochs=quantum_epochs,
            hybrid_epochs=hybrid_epochs
        )

    # Save training history
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update checkpoint directory (use absolute path)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.checkpoint_dir = checkpoint_dir

    # Save training history
    trainer.save_history("training_history.json")
    if (Path(config.training.log_dir) / "training_history.json").exists():
        (output_dir / "training_history.json").write_text(
            (Path(config.training.log_dir) / "training_history.json").read_text()
        )

    # Plot training history
    plot_training_history(
        trainer.history,
        save_path=str(output_dir / "training_curves.png")
    )

    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Test Evaluation")
    print("="*50)

    test_metrics = trainer.evaluate(test_loader)
    print_metrics(test_metrics, "Test Set Metrics")

    # Save final model
    final_checkpoint = output_dir / "checkpoints" / "final_model.pt"
    trainer.save_checkpoint("final_model.pt")
    print(f"\nModel saved to {final_checkpoint}")

    print("\n" + "="*50)
    print("Training Completed Successfully!")
    print("="*50)


def evaluate(args, config):
    """Run evaluation pipeline."""
    print("\n" + "="*50)
    print("Q-GAD Evaluation Pipeline")
    print("="*50)

    if args.checkpoint is None:
        raise ValueError("Must specify --checkpoint for evaluation")

    # Set seed
    set_seed(args.seed if args.seed else config.seed)

    # Get device
    device = get_device(args.device if args.device else config.training.device)

    # Load dataset
    dataset = load_dataset(args, config)
    _, _, test_loader = setup_data_loaders(dataset, config)

    # Create model
    n_modes = config.gbs.n_modes
    classical_dim = dataset.node_features.shape[1] if dataset.node_features is not None else 10

    model, trainer = create_model_and_trainer(
        n_modes=n_modes,
        classical_feature_dim=classical_dim,
        device=device,
        use_parallel=args.use_parallel,
        use_ddp=args.use_ddp
    )

    # Load checkpoint
    trainer.load_checkpoint(args.checkpoint)

    # Evaluate
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_loader)
    print_metrics(test_metrics, "Test Set Metrics")

    # Generate predictions and plot
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in test_loader:
        squeezing = batch["squeezing"].to(device)
        unitary = batch["unitary"].to(device)
        classical = batch["classical_features"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            logits = model(squeezing, unitary, classical)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    # Plot confusion matrix
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        np.array(all_labels),
        np.array(all_preds),
        save_path=str(output_dir / "confusion_matrix.png")
    )

    print("\nEvaluation completed!")


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    if args.config:
        config = ExperimentConfig.load(args.config)
    else:
        if args.preset == "testing":
            config = get_testing_config()
        elif args.preset == "production":
            from src.configs.config import get_production_config
            config = get_production_config()
        else:
            config = ExperimentConfig()

    # Override config with command line arguments
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    if args.use_synthetic:
        config.data.use_synthetic = True

    # Save config
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(output_dir / "config.json"))

    # Run appropriate mode
    if args.mode == "train":
        train(args, config)
    elif args.mode == "evaluate":
        evaluate(args, config)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
