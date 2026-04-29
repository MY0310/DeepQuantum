"""Run a faster Q-GAD training pass on Elliptic++ for smoke/perf checks."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader, Subset
from data.financial_dataset import load_elliptic_dataset, collate_fn
from trainer import create_model_and_trainer
from utils.helpers import set_seed, get_device, print_metrics
from visualization.common_plots import plot_training_history


def main():
    # Configuration
    set_seed(42)
    device = get_device()

    print("="*60)
    print("Q-GAD Training on Elliptic++ (FAST TEST VERSION)")
    print("="*60)
    print("\n[Fast Mode]")
    print()

    # Load Elliptic++ dataset
    print("Loading Elliptic++ dataset...")
    train_dataset, test_dataset = load_elliptic_dataset(
        data_dir=str(PROJECT_ROOT / "data" / "elliptic"),
        max_nodes=20,
        ego_radius=1.5,
        train_periods=(1, 34),
        test_periods=(35, 49),
        cache_dir=str(PROJECT_ROOT / "data" / "elliptic" / "processed")
    )

    print(f"\n完整数据集大小:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")

    # Fast-mode subset sizes
    train_size = min(20000, len(train_dataset))
    val_size = min(4000, len(train_dataset))
    test_size = min(4000 , len(test_dataset))

    print(f"\n快速测试数据集:")
    print(f"  Train samples: {train_size}")
    print(f"  Val samples: {val_size}")
    print(f"  Test samples: {test_size}")

    # Create subsets
    train_subset = Subset(train_dataset, range(train_size))
    val_subset = Subset(train_dataset, range(train_size, train_size + val_size))
    test_subset = Subset(test_dataset, range(test_size))

    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=32,  # 适中的 batch size
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
        test_subset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    print(f"\nDataLoader 配置:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Create model
    print("\nCreating model...")
    classical_dim = train_dataset.node_features.shape[1]
    model, trainer = create_model_and_trainer(
        n_modes=10,  # Reduced for faster smoke/performance checks
        classical_feature_dim=classical_dim,
        device=device,
        use_parallel=False
    )
    trainer.checkpoint_dir = PROJECT_ROOT / "checkpoints"
    trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.log_dir = PROJECT_ROOT / "logs"
    trainer.log_dir.mkdir(parents=True, exist_ok=True)
    trainer.setup_optimizers(quantum_lr=1e-3, classifier_lr=1e-3)

    # Train
    print("\n" + "="*60)
    print("Starting Training (快速模式)")
    print("="*60)

    # Enforce real DeepQuantum backend (no mock fallback allowed)
    has_real_backend = bool(getattr(model.quantum_extractor.gbs_kernel, "has_deepquantum", False))
    print(f"Real DeepQuantum backend: {has_real_backend}")
    if not has_real_backend:
        raise RuntimeError(
            "DeepQuantum backend unavailable. This script forbids mock backend. "
            "Please run in an environment with deepquantum installed."
        )

    print("Using DeepQuantum for real quantum simulation")
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
    output_dir = PROJECT_ROOT / "outputs" / "elliptic_fast"
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_checkpoint("elliptic_fast_model.pt")
    trainer.save_history("elliptic_fast_history.json")

    # Plot training curves
    plot_training_history(
        trainer.history,
        save_path=str(output_dir / "training_curves.png")
    )

    print(f"\nResults saved to {output_dir}")

    print("\n" + "="*60)
    print("[Done] Fast run finished.")
    print("="*60)
    print("\n如果结果正常，可以运行完整版本:")
    print("  python run_elliptic.py")
    print()


if __name__ == "__main__":
    main()
