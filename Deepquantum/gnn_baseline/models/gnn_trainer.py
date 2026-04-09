"""
Trainer for Classical GNN Baseline Models

This module provides a training pipeline that mirrors the Q-GAD training process,
ensuring fair comparison between quantum and classical approaches.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.financial_dataset import collate_fn


class GNNTrainer:
    """
    Trainer for classical GNN models.

    Training strategy:
    - End-to-end training (vs alternating optimization in Q-GAD)
    - Same evaluation metrics for fair comparison
    - Compatible with Elliptic dataset format
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "./gnn_baseline/checkpoints",
        log_dir: str = "./gnn_baseline/logs"
    ):
        """
        Initialize GNN trainer.

        Args:
            model: ClassicalGNNSystem instance
            device: Training device
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization
            checkpoint_dir: Directory for model checkpoints
            log_dir: Directory for training logs
        """
        self.device = torch.device(device)
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
            "val_auc": [],
            "val_precision": [],
            "val_recall": []
        }

        # Optimizer and loss
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Average loss and accuracy
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch (quantum format: squeezing, unitary, classical_features, label)
            squeezing = batch_data['squeezing'].to(self.device)  # [batch, n_modes]
            unitary = batch_data['unitary'].to(self.device)      # [batch, n_modes, n_modes]
            classical_features = batch_data['classical_features'].to(self.device)
            labels = batch_data['label'].to(self.device)

            batch_size = squeezing.size(0)
            n_modes = squeezing.size(1)

            # Reconstruct adjacency from unitary (for GNN)
            # Unitary represents the interferometer, we can use it as weighted adjacency
            adj_matrices = torch.abs(unitary)  # [batch, n_modes, n_modes]

            # Node features: use squeezing as base features
            node_features = squeezing.unsqueeze(-1).expand(-1, -1, classical_features.size(1))

            # Convert to PyTorch Geometric format
            edge_indices = []
            batches = []
            node_offset = 0

            for i in range(batch_size):
                # Extract single graph from batch
                adj = adj_matrices[i]
                node_feat = node_features[i]

                # Create edge_index from weighted adjacency
                # Use threshold to sparsify (keep top edges)
                edge_threshold = adj.mean() + 0.5 * adj.std()
                edge_index = (adj > edge_threshold).nonzero().t()

                # Add offset for batching
                edge_index = edge_index + node_offset
                node_offset += node_feat.size(0)

                edge_indices.append(edge_index)
                batches.append(torch.full((node_feat.size(0),), i, device=self.device))

            # Concatenate all nodes
            all_node_features = node_features.view(-1, node_features.size(-1))
            all_edge_index = torch.cat(edge_indices, dim=1)
            all_batch = torch.cat(batches, dim=0)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(
                all_node_features,
                all_edge_index,
                all_batch,
                classical_features
            )

            # Compute loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_data in tqdm(val_loader, desc="Evaluating"):
                # Unpack batch (quantum format)
                squeezing = batch_data['squeezing'].to(self.device)
                unitary = batch_data['unitary'].to(self.device)
                classical_features = batch_data['classical_features'].to(self.device)
                labels = batch_data['label'].to(self.device)

                batch_size = squeezing.size(0)
                n_modes = squeezing.size(1)

                # Reconstruct adjacency from unitary
                adj_matrices = torch.abs(unitary)
                node_features = squeezing.unsqueeze(-1).expand(-1, -1, classical_features.size(1))

                # Convert to PyTorch Geometric format
                edge_indices = []
                batches = []
                node_offset = 0

                for i in range(batch_size):
                    adj = adj_matrices[i]
                    # Use threshold to sparsify
                    edge_threshold = adj.mean() + 0.5 * adj.std()
                    edge_index = (adj > edge_threshold).nonzero().t()
                    edge_index = edge_index + node_offset
                    node_offset += node_features[i].size(0)

                    edge_indices.append(edge_index)
                    batches.append(torch.full((node_features[i].size(0),), i, device=self.device))

                all_node_features = node_features.view(-1, node_features.size(-1))
                all_edge_index = torch.cat(edge_indices, dim=1)
                all_batch = torch.cat(batches, dim=0)

                # Forward pass
                logits = self.model(
                    all_node_features,
                    all_edge_index,
                    all_batch,
                    classical_features
                )

                # Compute loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                # Predictions
                probs = torch.softmax(logits, dim=1)
                pred = logits.argmax(dim=1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * (all_preds == all_labels).mean(),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'auc': roc_auc_score(all_labels, all_probs),
            'ap': average_precision_score(all_labels, all_probs)
        }

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        metrics['confusion_matrix'] = cm.tolist()

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 10,
        early_stopping_patience: int = 5
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
        """
        print("=" * 60)
        print("Training Classical GNN Model")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {n_epochs}")
        print(f"Optimizer: Adam (lr={self.optimizer.param_groups[0]['lr']})")
        print("=" * 60)

        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(1, n_epochs + 1):
            print(f"\nEpoch {epoch}/{n_epochs}")
            print("-" * 60)

            # Train
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.evaluate(val_loader)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])

            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"Val AUC: {val_metrics['auc']:.4f} | Val F1: {val_metrics['f1']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])

            # Early stopping based on AUC
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0
                self.save_checkpoint(f"best_model_epoch_{epoch}.pt")
                print(f"✓ New best model saved (AUC: {best_val_auc:.4f})")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break

        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)
        print(f"Best Validation AUC: {best_val_auc:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def save_history(self, filename: str):
        """Save training history to JSON."""
        path = self.log_dir / filename
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")
