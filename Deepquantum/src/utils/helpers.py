"""
Utility functions for Q-GAD system.

Includes:
- Device management
- Random seed setting
- Metrics computation
"""

import os
import sys
import random
from pathlib import Path
from typing import Dict, Optional, Tuple


def _patch_windows_conda_dll_path() -> None:
    """
    Ensure conda DLL lookup paths are present when Python is launched
    directly via `...\\envs\\qgad\\python.exe` without shell activation.
    """
    if os.name != "nt":
        return

    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        exe_parent = Path(sys.executable).resolve().parent
        if (exe_parent / "conda-meta").exists():
            prefix = str(exe_parent)
    if not prefix:
        return

    dll_dirs = [
        Path(prefix),
        Path(prefix) / "Library" / "mingw-w64" / "bin",
        Path(prefix) / "Library" / "usr" / "bin",
        Path(prefix) / "Library" / "bin",
        Path(prefix) / "Scripts",
    ]
    existing = [str(p) for p in dll_dirs if p.exists()]
    if not existing:
        return

    parts = [p for p in os.environ.get("PATH", "").split(";") if p]
    for p in reversed(existing):
        if p not in parts:
            parts.insert(0, p)
    os.environ["PATH"] = ";".join(parts)

    add_dll = getattr(os, "add_dll_directory", None)
    if add_dll is not None:
        for p in existing:
            try:
                add_dll(p)
            except OSError:
                pass


_patch_windows_conda_dll_path()

import torch
import numpy as np


def _configure_matplotlib_runtime(per_process: bool = True) -> Path:
    """
    Configure matplotlib cache dir to a writable project-local path.

    This avoids Windows permission/lock issues at:
    `C:\\Users\\...\\.matplotlib\\fontlist-*.json.matplotlib-lock`
    especially under multi-process workloads.
    """
    project_root = Path(__file__).resolve().parents[2]
    mpl_root = project_root / ".mplconfig"
    mpl_root.mkdir(parents=True, exist_ok=True)

    if per_process:
        mpl_dir = mpl_root / f"pid_{os.getpid()}"
    else:
        mpl_dir = mpl_root
    mpl_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    return mpl_dir


# Configure once at module import for the current process.
_configure_matplotlib_runtime(per_process=True)


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
    def _best_available() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if device is not None:
        requested = str(device).strip().lower()
        if requested == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            fallback = _best_available()
            print(f"[Device] CUDA requested but unavailable, fallback to '{fallback}'.")
            return fallback
        if requested == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            fallback = _best_available()
            print(f"[Device] MPS requested but unavailable, fallback to '{fallback}'.")
            return fallback
        return requested

    return _best_available()


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


def logits_to_binary_predictions(
    logits: torch.Tensor,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert binary-class logits into fraud probability and thresholded labels.

    Args:
        logits: Model logits with shape [batch, 2]
        threshold: Decision threshold applied on fraud probability

    Returns:
        probs_pos: Fraud-class probabilities [batch]
        preds: Thresholded predictions {0,1} [batch]
    """
    probs = torch.softmax(logits, dim=1)
    probs_pos = probs[:, 1]
    preds = (probs_pos >= float(threshold)).long()
    return probs_pos, preds


def find_best_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_threshold: float = 0.01,
    max_threshold: float = 0.99,
    num_thresholds: int = 199,
) -> Dict[str, float]:
    """
    Grid-search threshold by maximizing F1 on validation data.

    Args:
        y_true: True binary labels
        y_prob: Fraud probabilities
        min_threshold: Minimum threshold to test
        max_threshold: Maximum threshold to test
        num_thresholds: Number of grid points

    Returns:
        Dictionary with best threshold and related metrics.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thresholds = np.linspace(min_threshold, max_threshold, num_thresholds)

    best = {
        "threshold": 0.5,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if f1 > best["f1"]:
            best = {
                "threshold": float(t),
                "f1": f1,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            }
    return best


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


def load_qgad_checkpoint_model(
    checkpoint_path: str,
    device: str = "cpu",
    n_modes: int = 20,
    n_shots: Optional[int] = None,
):
    """
    Rebuild QGADSystem from checkpoint weights by inferring architecture dims.

    This prevents strict shape mismatch when hidden dims differ between
    historical checkpoints and current defaults.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Infer classical input dimension from first classical linear layer.
    c_in = 166
    for k, v in state_dict.items():
        if "hybrid_classifier.classical_encoder.encoder" in k and k.endswith(".weight") and v.ndim == 2:
            c_in = int(v.shape[1])
            break

    # Infer hidden dims from classical encoder linear layer outputs.
    layer_shapes = []
    for k, v in state_dict.items():
        if "hybrid_classifier.classical_encoder.encoder" in k and k.endswith(".weight") and v.ndim == 2:
            try:
                idx = int(k.split(".")[3])  # ...encoder.{idx}.weight
            except (IndexError, ValueError):
                continue
            layer_shapes.append((idx, int(v.shape[0])))
    layer_shapes.sort(key=lambda x: x[0])
    hidden_dims = [d for _, d in layer_shapes] if layer_shapes else [64, 32]
    # FeatureEncoder appends a final projection to hidden_dims[-1], which appears
    # as a duplicated trailing output dim in checkpoint linear layers.
    if len(hidden_dims) >= 2 and hidden_dims[-1] == hidden_dims[-2]:
        hidden_dims = hidden_dims[:-1]

    # Infer quantum feature input dim.
    q_in = 9
    for k, v in state_dict.items():
        if "hybrid_classifier.quantum_encoder.encoder" in k and k.endswith(".weight") and v.ndim == 2:
            q_in = int(v.shape[1])
            break

    from gbs.gbs_kernel import GBSConfig
    from models.hybrid_classifier import HybridConfig, QGADSystem

    # Re-assert per-process MPL runtime before creating the model, because
    # DeepQuantum import path may trigger matplotlib font-cache operations.
    _configure_matplotlib_runtime(per_process=True)

    gbs_config = GBSConfig(
        n_modes=n_modes,
        n_shots=int(n_shots) if n_shots is not None else 100,
        backend="gaussian",
        use_displacement=True,
        device=device,
    )
    hybrid_config = HybridConfig(
        quantum_feature_dim=q_in,
        classical_feature_dim=c_in,
        hidden_dims=hidden_dims,
        dropout=0.1,
        device=device,
    )
    model = QGADSystem(
        gbs_config=gbs_config,
        hybrid_config=hybrid_config,
        use_xgboost_fusion=False,
    )
    incompatible = model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    if n_shots is not None:
        model.quantum_extractor.gbs_kernel.config.n_shots = int(n_shots)

    if len(incompatible.missing_keys) > 0 or len(incompatible.unexpected_keys) > 0:
        print(
            "[Checkpoint] Non-strict load: "
            f"missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)}"
        )

    return model, checkpoint, {
        "classical_feature_dim": c_in,
        "quantum_feature_dim": q_in,
        "hidden_dims": hidden_dims,
    }


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
