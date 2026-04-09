"""
GNN Baseline Experiment for Q-GAD Comparison

This script runs classical GNN models on the Elliptic++ dataset to provide
fair comparison with the quantum GBS approach.

Usage:
    python run_gnn_baseline.py --model gcn --epochs 10
    python run_gnn_baseline.py --model gat --epochs 20 --fast
    python run_gnn_baseline.py --model all  # Run all GNN types
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from torch.utils.data import DataLoader, Subset

# Import from parent project
from data.financial_dataset import load_elliptic_dataset, collate_fn
from utils.helpers import set_seed, get_device, print_metrics, plot_training_history

# Import GNN models
sys.path.insert(0, str(Path(__file__).parent / "models"))
from gnn_models import GNNConfig, create_gnn_model
from gnn_trainer import GNNTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GNN baseline models for Q-GAD comparison'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gcn',
        choices=['gcn', 'gat', 'sage', 'gin', 'all'],
        help='GNN architecture to train (default: gcn)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training (default: 32)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )

    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=64,
        help='Hidden dimension for GNN layers (default: 64)'
    )

    parser.add_argument(
        '--num-layers',
        type=int,
        default=3,
        help='Number of GNN layers (default: 3)'
    )

    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast mode: reduced dataset for quick testing'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, auto-detected if not specified)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    return parser.parse_args()


def train_single_model(model_type, args, train_loader, val_loader, test_loader, output_dir):
    """
    Train a single GNN model.

    Args:
        model_type: Type of GNN ('gcn', 'gat', 'sage', 'gin')
        args: Command line arguments
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        output_dir: Directory to save results

    Returns:
        Dictionary of results
    """
    print("\n" + "=" * 80)
    print(f"Training {model_type.upper()} Baseline Model")
    print("=" * 80)

    # Get classical feature dimension from dataset
    classical_dim = train_loader.dataset.dataset.node_features.shape[1]

    # Create model configuration
    config = GNNConfig(
        gnn_type=model_type,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        dropout=0.1,
        classical_feature_dim=classical_dim,
        device=args.device
    )

    # Create model
    model = create_gnn_model(model_type, config)

    # Print model info
    print(f"\nModel Configuration:")
    print(f"  Architecture: {model_type.upper()}")
    print(f"  Num Layers: {config.num_layers}")
    print(f"  Hidden Dim: {config.hidden_dim}")
    print(f"  Graph Feature Dim: {config.graph_feature_dim}")
    print(f"  Classical Feature Dim: {config.classical_feature_dim}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Create trainer
    trainer = GNNTrainer(
        model=model,
        device=args.device,
        learning_rate=args.lr,
        checkpoint_dir=str(output_dir / f"{model_type}_checkpoints"),
        log_dir=str(output_dir / f"{model_type}_logs")
    )

    # Train
    print(f"\n{'=' * 80}")
    print(f"Starting Training ({args.epochs} epochs)")
    print(f"{'=' * 80}")

    trainer.train(
        train_loader,
        val_loader,
        n_epochs=args.epochs,
        early_stopping_patience=5
    )

    # Evaluate on test set
    print(f"\n{'=' * 80}")
    print(f"Final Test Evaluation - {model_type.upper()}")
    print(f"{'=' * 80}")

    test_metrics = trainer.evaluate(test_loader)
    print_metrics(test_metrics, f"Test Set Metrics ({model_type.upper()})")

    # Save results
    model_output_dir = output_dir / model_type
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    trainer.save_checkpoint(f"{model_type}_best_model.pt")

    # Save training history
    trainer.save_history(f"{model_type}_history.json")

    # Plot training curves
    plot_training_history(
        trainer.history,
        save_path=str(model_output_dir / f"{model_type}_training_curves.png")
    )

    # Save test metrics
    with open(model_output_dir / f"{model_type}_test_metrics.json", 'w') as f:
        # Convert numpy arrays for JSON serialization
        metrics_serializable = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in test_metrics.items()
        }
        json.dump(metrics_serializable, f, indent=2)

    print(f"\nResults saved to {model_output_dir}")

    return {
        'model_type': model_type,
        'test_metrics': test_metrics,
        'total_params': total_params,
        'trainable_params': trainable_params
    }


def main():
    """Main experiment runner."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Setup device
    if args.device is None:
        args.device = get_device()
    device = torch.device(args.device)

    print("=" * 80)
    print("GNN Baseline Experiment for Q-GAD Comparison")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Random Seed: {args.seed}")
    print(f"Fast Mode: {args.fast}")
    print("=" * 80)

    # Load Elliptic++ dataset
    print("\nLoading Elliptic++ dataset...")
    train_dataset, test_dataset = load_elliptic_dataset(
        data_dir="./data/elliptic",
        max_nodes=20,
        ego_radius=1.5,
        train_periods=(1, 34),
        test_periods=(35, 49),
        cache_dir="./data/elliptic/processed"
    )

    # Subsample for fast mode
    if args.fast:
        train_size = min(1000, len(train_dataset))
        val_size = min(200, len(train_dataset))
        test_size = min(200, len(test_dataset))

        print(f"\n⚡ Fast Mode - Reduced Dataset:")
        print(f"  Train samples: {train_size}")
        print(f"  Val samples: {val_size}")
        print(f"  Test samples: {test_size}")

        # Need to create subsets from original dataset before wrapping
        train_subset = Subset(train_dataset, range(train_size))
        val_subset = Subset(train_dataset, range(train_size, train_size + val_size))
        test_subset = Subset(test_dataset, range(test_size))

        # Rename for consistency
        train_dataset = train_subset
        val_dataset = val_subset
        test_dataset = test_subset
    else:
        # Split train into train/val
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )

        print(f"\nDataset Statistics:")
        print(f"  Train samples: {train_size}")
        print(f"  Val samples: {val_size}")
        print(f"  Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"./gnn_baseline/outputs/experiment_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to train
    if args.model == 'all':
        model_types = ['gcn', 'gat', 'sage', 'gin']
    else:
        model_types = [args.model]

    # Train models
    all_results = {}

    for model_type in model_types:
        result = train_single_model(
            model_type,
            args,
            train_loader,
            val_loader,
            test_loader,
            output_dir
        )
        all_results[model_type] = result

    # Generate comparison report
    print("\n" + "=" * 80)
    print("Experiment Complete - Comparison Summary")
    print("=" * 80)

    print(f"\n{'Model':<10} {'Test AUC':<10} {'Test F1':<10} {'Params':<15}")
    print("-" * 80)

    for model_type, result in all_results.items():
        metrics = result['test_metrics']
        params = result['total_params']
        print(f"{model_type.upper():<10} "
              f"{metrics['auc']:<10.4f} "
              f"{metrics['f1']:<10.4f} "
              f"{params:<15,}")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'args': vars(args),
        'results': all_results
    }

    with open(output_dir / "experiment_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Experiment results saved to {output_dir}")
    print("\nNext steps:")
    print("  1. Compare with quantum results using analysis/generate_comparison_report.py")
    print("  2. Visualize differences using analysis/differential_analysis.py")


if __name__ == "__main__":
    import numpy as np  # Import at top level
    main()
