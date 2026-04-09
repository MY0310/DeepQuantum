"""
Run Q-GAD training on Elliptic++ Bitcoin dataset.

This script provides a complete pipeline for:
1. Downloading/loading Elliptic++ dataset
2. Preprocessing with quantum encoding
3. Training with alternating optimization
4. Evaluation and visualization
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from torch.utils.data import DataLoader
from data.financial_dataset import load_elliptic_dataset, collate_fn
from trainer import create_model_and_trainer
from utils.helpers import set_seed, get_device, print_metrics


def main():
    # Configuration
    set_seed(42)
    device = get_device()

    print("="*60)
    print("Q-GAD Training on Elliptic++ Bitcoin Dataset")
    print("="*60)

    # Load Elliptic++ dataset
    print("\nLoading Elliptic++ dataset...")
    train_dataset, test_dataset = load_elliptic_dataset(
        data_dir="./data/elliptic",
        max_nodes=20,
        ego_radius=1.5,
        train_periods=(1, 34),    # Periods 1-34 for training
        test_periods=(35, 49),    # Periods 35-49 for testing
        cache_dir="./data/elliptic/processed"
    )

    print(f"\nDataset statistics:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Create model
    print("\nCreating model...")
    classical_dim = train_dataset.node_features.shape[1]
    model, trainer = create_model_and_trainer(
        n_modes=20,
        classical_feature_dim=classical_dim,
        device=device,
        use_parallel=False  # Set to True if using multiple GPUs
    )
    trainer.setup_optimizers(quantum_lr=1e-3, classifier_lr=1e-3)

    # Train
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)

    # Check if using mock implementation
    if not model.quantum_extractor.gbs_kernel.has_deepquantum:
        print("Note: Using mock quantum implementation")
        quantum_epochs = 0
        hybrid_epochs = 10
    else:
        quantum_epochs = 1
        hybrid_epochs = 1

    trainer.train_alternating(
        train_loader,
        val_loader,
        n_epochs=10,
        quantum_epochs=quantum_epochs,
        hybrid_epochs=hybrid_epochs
    )

    # Final test evaluation
    print("\n" + "="*60)
    print("Final Test Evaluation")
    print("="*60)

    test_metrics = trainer.evaluate(test_loader)
    print_metrics(test_metrics, "Test Set Metrics")

    # Save results
    output_dir = Path("./outputs/elliptic_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_checkpoint("elliptic_model.pt")
    trainer.save_history("elliptic_history.json")

    # Plot training curves
    from utils.helpers import plot_training_history
    plot_training_history(
        trainer.history,
        save_path=str(output_dir / "training_curves.png")
    )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
