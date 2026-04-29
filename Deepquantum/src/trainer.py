"""
Training and evaluation pipeline for Q-GAD system.

This module implements:
- Alternating training strategy (quantum kernel + hybrid classifier)
- Full training pipeline with XGBoost fusion
- Evaluation metrics for fraud detection
- Model checkpointing and logging
- GPU parallel training support
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix
)

import sys
sys.path.append(str(Path(__file__).parent))

from models.hybrid_classifier import (
    QGADSystem, XGBoostFusionClassifier, HybridConfig
)
from gbs.gbs_kernel import GBSConfig
from data.financial_dataset import FinancialGraphDataset, collate_fn
from utils.distributed import get_parallel_model, adjust_batch_size, get_device_for_rank


class QGADTrainer:
    """
    Trainer for Q-GAD system with alternating optimization.

    Training phases:
    1. Train quantum kernel (contrastive learning)
    2. Train hybrid classifier (supervised learning)
    3. Train XGBoost fusion (optional)
    """

    def __init__(
        self,
        model: QGADSystem,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints",
        log_dir: str = "./logs",
        use_parallel: bool = False,
        use_ddp: bool = False
    ):
        """
        Initialize trainer.

        Args:
            model: QGADSystem instance
            device: Training device
            checkpoint_dir: Directory for model checkpoints
            log_dir: Directory for training logs
            use_parallel: Whether to use multi-GPU training
            use_ddp: Whether to use DistributedDataParallel (vs DataParallel)
        """
        self.device = get_device_for_rank() if "RANK" in os.environ else torch.device(device)

        # Apply parallel wrapper if requested
        if use_parallel:
            self.model = get_parallel_model(model, use_ddp=use_ddp)
        else:
            self.model = model.to(self.device)

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "val_auc": []
        }
        self.best_state_dict = None

        # Optimizers
        self.quantum_optimizer = None
        self.classifier_optimizer = None

    def _get_base_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _uses_real_deepquantum(self) -> bool:
        try:
            base = self._get_base_model()
            return bool(getattr(base.quantum_extractor.gbs_kernel, "has_deepquantum", False))
        except Exception:
            return False

    def _quantum_inputs_on_cpu(self) -> bool:
        """
        For real DeepQuantum path, keep squeezing/unitary on CPU to avoid
        wasteful GPU->CPU round-trips inside gbs_kernel forward.
        """
        is_single_device = not hasattr(self.model, "module")
        return is_single_device and str(self.device).startswith("cuda") and self._uses_real_deepquantum()

    def setup_optimizers(
        self,
        quantum_lr: float = 1e-3,
        classifier_lr: float = 1e-3
    ):
        """Setup optimizers for alternating training."""
        self.quantum_optimizer = torch.optim.Adam(
            self.model.quantum_extractor.parameters(),
            lr=quantum_lr
        )

        self.classifier_optimizer = torch.optim.Adam(
            self.model.hybrid_classifier.parameters(),
            lr=classifier_lr
        )

    def train_epoch(
        self,
        train_loader,
        phase: str = "hybrid",
        quantum_batches: int = 10
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            phase: 'quantum' or 'hybrid'
            quantum_batches: Number of batches to train quantum kernel

        Returns:
            Metrics dictionary
        """
        if phase == "quantum":
            return self._train_quantum_kernel(train_loader, quantum_batches)
        else:
            return self._train_hybrid_classifier(train_loader)

    def _train_quantum_kernel(
        self,
        train_loader,
        n_batches: int = 10
    ) -> Dict[str, float]:
        """Train quantum kernel with contrastive learning."""
        self.model.train()
        self.model.training_phase = "quantum"

        total_loss = 0.0
        n_processed = 0

        pbar = tqdm(train_loader, desc="Training Quantum Kernel")

        for batch in pbar:
            if n_processed >= n_batches:
                break

            # Move to device
            if self._quantum_inputs_on_cpu():
                squeezing = batch["squeezing"]
                unitary = batch["unitary"]
            else:
                squeezing = batch["squeezing"].to(self.device, non_blocking=True)
                unitary = batch["unitary"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # Train quantum kernel
            metrics = self.model.train_quantum_kernel(
                squeezing, unitary, labels, self.quantum_optimizer
            )

            total_loss += metrics["loss"]
            n_processed += 1

            pbar.set_postfix({"loss": f"{metrics['loss']:.4f}"})

        avg_loss = total_loss / max(n_processed, 1)

        return {"loss": avg_loss}

    def _train_hybrid_classifier(self, train_loader) -> Dict[str, float]:
        """Train hybrid classifier with supervised learning."""
        self.model.train()
        self.model.training_phase = "hybrid"

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc="Training Hybrid Classifier")

        for batch in pbar:
            # Move to device
            if self._quantum_inputs_on_cpu():
                squeezing = batch["squeezing"]
                unitary = batch["unitary"]
            else:
                squeezing = batch["squeezing"].to(self.device, non_blocking=True)
                unitary = batch["unitary"].to(self.device, non_blocking=True)
            classical = batch["classical_features"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # Train classifier
            metrics = self.model.train_hybrid_classifier(
                squeezing, unitary, classical, labels, self.classifier_optimizer
            )

            total_loss += metrics["loss"]
            total_acc += metrics["accuracy"]
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "acc": f"{metrics['accuracy']:.4f}"
            })

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches

        return {"loss": avg_loss, "accuracy": avg_acc}

    @torch.no_grad()
    def evaluate(self, val_loader) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Evaluation metrics
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            if self._quantum_inputs_on_cpu():
                squeezing = batch["squeezing"]
                unitary = batch["unitary"]
            else:
                squeezing = batch["squeezing"].to(self.device, non_blocking=True)
                unitary = batch["unitary"].to(self.device, non_blocking=True)
            classical = batch["classical_features"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            # Forward pass
            logits = self.model(squeezing, unitary, classical)

            # Compute loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = {
            "loss": total_loss / n_batches,
            "accuracy": float(np.mean(all_preds == all_labels)),
            "precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "recall": float(recall_score(all_labels, all_preds, zero_division=0)),
            "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
            "auc": float(roc_auc_score(all_labels, all_probs)),
            "ap": float(average_precision_score(all_labels, all_probs))
        }

        return metrics

    def train_alternating(
        self,
        train_loader,
        val_loader,
        n_epochs: int = 50,
        quantum_epochs: int = 1,
        hybrid_epochs: int = 1
    ):
        """
        Train with alternating optimization.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Total number of epochs
            quantum_epochs: Epochs of quantum training per cycle
            hybrid_epochs: Epochs of hybrid training per cycle
        """
        best_f1 = 0.0

        for epoch in range(n_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{n_epochs}")
            print(f"{'='*50}")

            # Phase 1: Train quantum kernel (skip if quantum_epochs=0)
            if quantum_epochs > 0:
                for q_epoch in range(quantum_epochs):
                    print(f"\n[Quantum Training {q_epoch + 1}/{quantum_epochs}]")
                    q_metrics = self.train_epoch(train_loader, phase="quantum")
                    print(f"Quantum Loss: {q_metrics['loss']:.4f}")

            # Phase 2: Train hybrid classifier
            for h_epoch in range(hybrid_epochs):
                print(f"\n[Hybrid Training {h_epoch + 1}/{hybrid_epochs}]")
                h_metrics = self.train_epoch(train_loader, phase="hybrid")
                print(f"Hybrid Loss: {h_metrics['loss']:.4f}, Acc: {h_metrics['accuracy']:.4f}")

            # Evaluate
            print(f"\n[Validation]")
            val_metrics = self.evaluate(val_loader)

            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")

            # Log history
            self.history["train_loss"].append(h_metrics["loss"])
            self.history["train_acc"].append(h_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_auc"].append(val_metrics["auc"])

            # Save checkpoint
            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                self.best_state_dict = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                print(f"Tracked best model (F1: {best_f1:.4f})")

        print(f"\nTraining completed! Best F1: {best_f1:.4f}")
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

    def train_with_xgboost_fusion(
        self,
        train_loader,
        val_loader,
        neural_epochs: int = 20
    ) -> XGBoostFusionClassifier:
        """
        Train neural network first, then extract features for XGBoost.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            neural_epochs: Epochs for neural network training

        Returns:
            Trained XGBoost classifier
        """
        # Phase 1: Train neural network
        print(f"[Phase 1] Training Neural Network ({neural_epochs} epochs)")
        self.train_alternating(
            train_loader, val_loader,
            n_epochs=neural_epochs,
            quantum_epochs=1,
            hybrid_epochs=1
        )

        # Phase 2: Extract features
        print(f"\n[Phase 2] Extracting Features for XGBoost")
        train_quantum, train_classical, train_labels = self._extract_features(
            train_loader
        )
        val_quantum, val_classical, val_labels = self._extract_features(
            val_loader
        )

        # Phase 3: Train XGBoost
        print(f"\n[Phase 3] Training XGBoost")
        xgb_clf = XGBoostFusionClassifier(
            quantum_feature_dim=train_quantum.shape[1],
            classical_feature_dim=train_classical.shape[1]
        )

        xgb_clf.fit(
            train_quantum, train_classical, train_labels,
            validation_data=(val_quantum, val_classical, val_labels)
        )

        # Evaluate XGBoost
        val_preds = xgb_clf.predict(val_quantum, val_classical)
        val_probs = xgb_clf.predict_proba(val_quantum, val_classical)[:, 1]

        print(f"\n[XGBoost Results]")
        print(f"Accuracy: {np.mean(val_preds == val_labels):.4f}")
        print(f"F1: {f1_score(val_labels, val_preds):.4f}")
        print(f"AUC: {roc_auc_score(val_labels, val_probs):.4f}")

        return xgb_clf

    def _extract_features(self, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract quantum and classical features using trained model."""
        self.model.eval()

        quantum_features = []
        classical_features = []
        labels = []

        for batch in tqdm(data_loader, desc="Extracting features"):
            if self._quantum_inputs_on_cpu():
                squeezing = batch["squeezing"]
                unitary = batch["unitary"]
            else:
                squeezing = batch["squeezing"].to(self.device, non_blocking=True)
                unitary = batch["unitary"].to(self.device, non_blocking=True)
            classical = batch["classical_features"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            with torch.no_grad():
                # Extract quantum features
                q_feat = self.model.quantum_extractor(squeezing, unitary)

            quantum_features.append(q_feat.cpu().numpy())
            classical_features.append(classical.cpu().numpy())
            labels.append(label.cpu().numpy())

        return (
            np.vstack(quantum_features),
            np.vstack(classical_features),
            np.concatenate(labels)
        )

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        import os
        path = self.checkpoint_dir / filename
        # Ensure directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "quantum_optimizer": self.quantum_optimizer.state_dict() if self.quantum_optimizer else None,
            "classifier_optimizer": self.classifier_optimizer.state_dict() if self.classifier_optimizer else None,
            "history": self.history
        }, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.quantum_optimizer is not None and checkpoint.get("quantum_optimizer") is not None:
            self.quantum_optimizer.load_state_dict(checkpoint["quantum_optimizer"])
        if self.classifier_optimizer is not None and checkpoint.get("classifier_optimizer") is not None:
            self.classifier_optimizer.load_state_dict(checkpoint["classifier_optimizer"])
        self.history = checkpoint["history"]

        print(f"Loaded checkpoint from {path}")

    def save_history(self, filename: str = "training_history.json"):
        """Save training history to JSON."""
        path = self.log_dir / filename
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)


def create_model_and_trainer(
    n_modes: int = 20,
    classical_feature_dim: int = 10,
    device: str = None,
    use_xgboost: bool = False,
    use_parallel: bool = False,
    use_ddp: bool = False
) -> Tuple[QGADSystem, QGADTrainer]:
    """
    Helper function to create model and trainer.

    Args:
        n_modes: Number of quantum modes
        classical_feature_dim: Dimension of classical features
        device: Training device
        use_xgboost: Whether to use XGBoost fusion
        use_parallel: Whether to use multi-GPU parallel training
        use_ddp: Whether to use DistributedDataParallel (vs DataParallel)

    Returns:
        Model and trainer instances
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    # Create configurations
    gbs_config = GBSConfig(
        n_modes=n_modes,
        n_shots=100,
        backend="gaussian",
        use_displacement=True,
        device=device
    )

    hybrid_config = HybridConfig(
        quantum_feature_dim=9,
        classical_feature_dim=classical_feature_dim,
        hidden_dims=[64, 32],
        dropout=0.1,
        device=device
    )

    # Create model
    model = QGADSystem(
        gbs_config=gbs_config,
        hybrid_config=hybrid_config,
        use_xgboost_fusion=use_xgboost
    )

    # Create trainer
    trainer = QGADTrainer(
        model, device=device,
        use_parallel=use_parallel,
        use_ddp=use_ddp
    )
    trainer.setup_optimizers(quantum_lr=1e-3, classifier_lr=1e-3)

    return model, trainer


if __name__ == "__main__":
    from torch.utils.data import random_split

    print("Creating synthetic dataset...")
    dataset = FinancialGraphDataset(
        edge_list=[],  # Will create synthetic
        max_nodes=20,
        ego_radius=1.5
    )

    # For testing, use synthetic
    from data.financial_dataset import SyntheticFinancialDataset
    dataset = SyntheticFinancialDataset(
        num_nodes=100,
        fraud_ratio=0.1,
        max_nodes=20
    )

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn
    )

    print("Creating model and trainer...")
    model, trainer = create_model_and_trainer(
        n_modes=20,
        classical_feature_dim=10,
        device="cpu"  # Use CPU for testing
    )

    print("\nStarting training...")
    trainer.train_alternating(
        train_loader, val_loader,
        n_epochs=5,
        quantum_epochs=1,
        hybrid_epochs=1
    )

    print("\nTraining completed!")
