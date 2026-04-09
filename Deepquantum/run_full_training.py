"""
Comprehensive training script for Q-GAD with different datasets and modes.

Usage:
    python run_full_training.py --dataset elliptic --mode full
    python run_full_training.py --dataset synthetic --nodes 5000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import DataLoader
from data.financial_dataset import (
    load_elliptic_dataset,
    SyntheticFinancialDataset,
    collate_fn
)
from trainer import create_model_and_trainer
from utils.helpers import set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Q-GAD Full Training Pipeline")

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="synthetic",
        choices=["synthetic", "elliptic"],
        help="Dataset to use"
    )
    parser.add_argument("--data_dir", type=str, default="./data/elliptic")
    parser.add_argument("--nodes", type=int, default=1000, help="Nodes for synthetic data")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--quantum_lr", type=float, default=1e-3)
    parser.add_argument("--classifier_lr", type=float, default=1e-3)

    # Model
    parser.add_argument("--n_modes", type=int, default=20)
    parser.add_argument("--max_nodes", type=int, default=20)

    # Advanced
    parser.add_argument("--use_xgboost", action="store_true")
    parser.add_argument("--use_parallel", action="store_true")
    parser.add_argument("--device", type=str, default=None)

    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--experiment_name", type=str, default="qgad_full")

    return parser.parse_args()


def load_dataset(args):
    """Load dataset based on arguments."""
    if args.dataset == "elliptic":
        print("Loading Elliptic++ dataset...")
        train_ds, test_ds = load_elliptic_dataset(
            data_dir=args.data_dir,
            max_nodes=args.max_nodes,
            ego_radius=1.5,
            train_periods=(1, 34),
            test_periods=(35, 49)
        )

        # Split train into train/val
        train_size = int(0.8 * len(train_ds))
        val_size = len(train_ds) - train_size
        train_subset, val_subset = torch.utils.data.random_split(
            train_ds, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        return train_subset, val_subset, test_ds

    else:  # synthetic
        print(f"Generating synthetic dataset ({args.nodes} nodes)...")
        dataset = SyntheticFinancialDataset(
            num_nodes=args.nodes,
            fraud_ratio=0.1,          # 10% fraud
            fraud_community_size=20,
            fraud_density=0.5,         # Dense fraud communities
            normal_density=0.05,       # Sparse normal connections
            max_nodes=args.max_nodes,
            ego_radius=1.5
        )

        # Split into train/val/test
        total = len(dataset)
        train_size = int(0.7 * total)
        val_size = int(0.15 * total)

        train_subset, val_subset, test_ds = torch.utils.data.random_split(
            dataset,
            [train_size, val_size, total - train_size - val_size],
            generator=torch.Generator().manual_seed(42)
        )

        return train_subset, val_subset, test_ds


def main():
    args = parse_args()

    # Setup
    set_seed(42)
    device = get_device(args.device)

    print("="*60)
    print("Q-GAD Full Training Pipeline")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    # Load data
    train_subset, val_subset, test_ds = load_dataset(args)

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_subset)}")
    print(f"  Val: {len(val_subset)}")
    print(f"  Test: {len(test_ds)}")

    # Data loaders
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    # Model
    print(f"\nCreating model ({args.n_modes} modes)...")
    classical_dim = 10 if args.dataset == "synthetic" else 166

    model, trainer = create_model_and_trainer(
        n_modes=args.n_modes,
        classical_feature_dim=classical_dim,
        device=device,
        use_xgboost=args.use_xgboost,
        use_parallel=args.use_parallel
    )
    trainer.setup_optimizers(
        quantum_lr=args.quantum_lr,
        classifier_lr=args.classifier_lr
    )

    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    # Determine training mode
    if not model.quantum_extractor.gbs_kernel.has_deepquantum:
        print("Using mock implementation - classifier only")
        quantum_epochs = 0
        hybrid_epochs = args.epochs
    else:
        quantum_epochs = 1
        hybrid_epochs = 1

    if args.use_xgboost:
        trainer.train_with_xgboost_fusion(
            train_loader, val_loader,
            neural_epochs=args.epochs
        )
    else:
        trainer.train_alternating(
            train_loader, val_loader,
            n_epochs=args.epochs,
            quantum_epochs=quantum_epochs,
            hybrid_epochs=hybrid_epochs
        )

    # Test evaluation
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)

    test_metrics = trainer.evaluate(test_loader)
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")
    print(f"Test AUC: {test_metrics['auc']:.4f}")

    # Save
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_checkpoint("final_model.pt")
    trainer.save_history("training_history.json")

    print(f"\nResults saved to {output_dir}")
    print("Training completed!")


if __name__ == "__main__":
    main()
