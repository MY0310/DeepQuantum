"""
Reusable plotting utilities for training/evaluation scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from visualization.mpl_setup import apply_default_style, get_plot_modules

ROOT = Path(__file__).resolve().parents[1]
plt = get_plot_modules(ROOT)
apply_default_style(plt)


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    if "train_loss" in history and "val_loss" in history:
        axes[0].plot(history["train_loss"], label="Train")
        axes[0].plot(history["val_loss"], label="Validation")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss")
        axes[0].legend()

    if "train_acc" in history and "val_acc" in history:
        axes[1].plot(history["train_acc"], label="Train")
        axes[1].plot(history["val_acc"], label="Validation")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy")
        axes[1].legend()

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
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))

    try:
        import seaborn as sns

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Fraud"],
            yticklabels=["Normal", "Fraud"],
        )
    except Exception:
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["Normal", "Fraud"])
        plt.yticks(tick_marks, ["Normal", "Fraud"])
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
):
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()
        plt.close()

