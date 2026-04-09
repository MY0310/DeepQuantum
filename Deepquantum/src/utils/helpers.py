"""
Utility functions for Q-GAD system.

Includes:
- Device management
- Random seed setting
- Metrics computation
- Visualization utilities
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from pathlib import Path

# Make seaborn optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed}")


def get_device(device: Optional[str] = None) -> str:
    """
    Get the best available device.

    Priority: CUDA > MPS > CPU

    Args:
        device: User-specified device (None = auto-detect)

    Returns:
        Device string
    """
    if device is not None:
        return device

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC)

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        accuracy_score, roc_auc_score, confusion_matrix
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            # Only one class present
            metrics["auc"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["true_negatives"] = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    metrics["false_positives"] = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    metrics["false_negatives"] = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    metrics["true_positives"] = int(cm[1, 1]) if cm.shape == (2, 2) else 0

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Print metrics in a formatted way.

    Args:
        metrics: Dictionary of metrics
        title: Section title
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None
):
    """
    Plot training history.

    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    if "train_loss" in history and "val_loss" in history:
        axes[0].plot(history["train_loss"], label="Train")
        axes[0].plot(history["val_loss"], label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()

    # Accuracy
    if "train_acc" in history and "val_acc" in history:
        axes[1].plot(history["train_acc"], label="Train")
        axes[1].plot(history["val_acc"], label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()

    # F1 / AUC
    if "val_f1" in history:
        axes[2].plot(history["val_f1"], label="F1", marker="o")
    if "val_auc" in history:
        axes[2].plot(history["val_auc"], label="AUC", marker="s")

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Validation Metrics")
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    if HAS_SEABORN:
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"]
        )
    else:
        # Fallback without seaborn
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Normal", "Fraud"])
        plt.yticks(tick_marks, ["Normal", "Fraud"])

        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
    """
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def create_checkpoint_dir(base_dir: str, experiment_name: str) -> Path:
    """
    Create checkpoint directory for experiment.

    Args:
        base_dir: Base directory
        experiment_name: Experiment name

    Returns:
        Path to checkpoint directory
    """
    path = Path(base_dir) / experiment_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_model(
    model: torch.nn.Module,
    path: str,
    metadata: Optional[Dict] = None
):
    """
    Save model with optional metadata.

    Args:
        model: PyTorch model
        path: Save path
        metadata: Optional metadata to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
    }

    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(
    model: torch.nn.Module,
    path: str,
    device: str = "cpu"
) -> Dict:
    """
    Load model from checkpoint.

    Args:
        model: PyTorch model instance
        path: Checkpoint path
        device: Device to load to

    Returns:
        Metadata dictionary (if any)
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {path}")

    return checkpoint.get("metadata", {})


class EarlyStopping:
    """
    Early stopping utility.

    Stops training when validation metric doesn't improve.
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",  # 'max' for metrics like AUC, 'min' for loss
        min_delta: float = 0.0
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait
            mode: 'max' or 'min'
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if should stop training.

        Args:
            score: Current validation score

        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


if __name__ == "__main__":
    print("Testing utility functions...")

    # Test seed setting
    set_seed(42)

    # Test device detection
    device = get_device()
    print(f"Device: {device}")

    # Test metrics
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.2, 0.8, 0.6, 0.3, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, "Test Metrics")

    # Test early stopping
    early_stop = EarlyStopping(patience=3, mode="max")
    scores = [0.7, 0.72, 0.71, 0.70, 0.69]

    for i, score in enumerate(scores):
        should_stop = early_stop(score)
        print(f"Epoch {i+1}: Score={score:.3f}, Stop={should_stop}")

    print("\nUtility tests completed!")
