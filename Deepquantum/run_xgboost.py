"""
XGBoost Baseline Experiment for Q-GAD

This script implements the baseline experiment using only classical features (no graph
topology, no quantum features) to verify the necessity of graph structural information.

Experiment Setup:
- Input: 166-dimensional classical features from Elliptic++ dataset
- Model: XGBoost classifier
- Purpose: Establish baseline performance without graph or quantum information

Author: Q-GAD Research Team
Date: 2026-01-20
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json
import argparse
import pickle
import time
from datetime import datetime
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    print("Warning: XGBoost not available, using sklearn's GradientBoosting")
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGBOOST = False

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path_like: str) -> Path:
    """Resolve path with project-root preference while tolerating cwd-style inputs."""
    p = Path(path_like)
    if p.is_absolute():
        return p

    cwd_candidate = (Path.cwd() / p).resolve()
    project_candidate = (PROJECT_ROOT / p).resolve()

    # For existing inputs (e.g., data dir), honor whichever exists.
    if project_candidate.exists():
        return project_candidate
    if cwd_candidate.exists():
        return cwd_candidate

    # For new outputs, keep paths under project root by default.
    p = project_candidate
    return p


class ClassicalOnlyBaseline:
    """
    XGBoost classifier using only classical node features.

    This baseline uses NO graph topology and NO quantum features,
    serving as the lower bound for performance comparison.
    """

    def __init__(self, random_state=42, **xgb_params):
        """
        Initialize classical-only baseline.

        Args:
            random_state: Random seed for reproducibility
            **xgb_params: Additional XGBoost parameters
        """
        self.random_state = random_state

        # Default XGBoost parameters optimized for imbalanced classification
        default_params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1.0,  # Will be adjusted for class imbalance
            "random_state": random_state,
            "eval_metric": "logloss"
        }
        default_params.update(xgb_params)

        if HAS_XGBOOST:
            self.model = xgb.XGBClassifier(**default_params)
            print("Initialized XGBoost classifier")
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=random_state
            )
            print("Initialized sklearn GradientBoosting classifier")

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train XGBoost on classical features only.

        Args:
            X_train: Training features [n_samples, 166]
            y_train: Training labels [n_samples]
            X_val: Optional validation features
            y_val: Optional validation labels
            verbose: Whether to print training progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training Classical-Only Baseline")
            print(f"{'='*60}")
            print(f"Training samples: {len(X_train)}")
            print(f"Feature dimension: {X_train.shape[1]}")

            # Class distribution
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"\nClass distribution:")
            for label, count in zip(unique, counts):
                ratio = count / len(y_train) * 100
                print(f"  Class {label}: {count:6d} ({ratio:5.2f}%)")

        start_time = time.time()

        # Adjust scale_pos_weight for class imbalance
        if HAS_XGBOOST:
            n_neg = np.sum(y_train == 0)
            n_pos = np.sum(y_train == 1)
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            self.model.set_params(scale_pos_weight=scale_pos_weight)

            if verbose:
                print(f"\nAdjusted scale_pos_weight: {scale_pos_weight:.4f}")

        # Train model
        if X_val is not None and y_val is not None and HAS_XGBOOST:
            # Try to use validation set (with or without early stopping)
            try:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=verbose
                )
            except TypeError:
                # If eval_set not supported, train without it
                self.model.fit(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)

        train_time = time.time() - start_time

        if verbose:
            print(f"\nTraining completed in {train_time:.2f} seconds")

        return self

    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluate model on test set.

        Args:
            X_test: Test features [n_samples, 166]
            y_test: Test labels [n_samples]
            verbose: Whether to print results

        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_proba),
            "ap": average_precision_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"Test Set Evaluation")
            print(f"{'='*60}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
            print(f"  AUC:       {metrics['auc']:.4f}")
            print(f"  AP:        {metrics['ap']:.4f}")
            print(f"\nConfusion Matrix:")
            cm = metrics['confusion_matrix']
            print(f"  [[{cm[0][0]:6d}, {cm[0][1]:6d}]")
            print(f"   [{cm[1][0]:6d}, {cm[1][1]:6d}]]")

        return metrics


def load_elliptic_data(data_dir="data/elliptic"):
    """
    Load Elliptic++ dataset directly from CSV files.

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    print(f"\n{'='*60}")
    print(f"Loading Elliptic++ Dataset")
    print(f"{'='*60}")

    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"

    # Load files
    features_path = raw_dir / "features.csv"
    classes_path = raw_dir / "classes.csv"
    edgelist_path = raw_dir / "edgelist.csv"

    if not features_path.exists() or not classes_path.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {raw_dir}. Expected files: features.csv, classes.csv"
        )

    print(f"Loading from {raw_dir}...")

    # Load features (first column is txId)
    print("  - Loading features...")
    features_df = pd.read_csv(features_path, index_col=0)
    print(f"    {len(features_df)} nodes, {features_df.shape[1]} features")

    # Load classes
    print("  - Loading classes...")
    classes_df = pd.read_csv(classes_path, index_col=0)
    print(f"    {len(classes_df)} labeled nodes")

    # Filter unknown nodes (class 0 or 'unknown')
    print("  - Filtering labeled nodes...")
    if classes_df['class'].dtype == object:
        labeled_mask = classes_df['class'] != 'unknown'
        classes_df['class'] = (
            classes_df['class']
            .map({'1': 1, '2': 2, 'unknown': 0})
            .fillna(0)
            .astype(int)
        )
    else:
        labeled_mask = classes_df['class'] != 0

    labeled_nodes = classes_df[labeled_mask].index
    common_nodes = labeled_nodes.intersection(features_df.index)

    features_df = features_df.loc[common_nodes]
    classes_df = classes_df.loc[common_nodes]

    # Convert labels to binary fraud target:
    # 1 -> 1 (illicit/fraud), 2 -> 0 (licit/normal)
    classes_df['label'] = classes_df['class'].map({1: 1, 2: 0})

    print(f"    Final: {len(features_df)} labeled nodes")
    print(f"      Licit (class 0): {(classes_df['label'] == 0).sum()}")
    print(f"      Illicit (class 1): {(classes_df['label'] == 1).sum()}")

    # Temporal split using first feature column as time
    print("  - Performing temporal split...")
    time_col = features_df.columns[0]

    train_mask = (features_df[time_col] >= 1) & (features_df[time_col] <= 34)
    test_mask = (features_df[time_col] >= 35) & (features_df[time_col] <= 49)

    train_indices = features_df[train_mask].index
    test_indices = features_df[test_mask].index

    # Extract train and test data
    X_train_full = features_df.loc[train_indices].values
    y_train_full = classes_df.loc[train_indices, 'label'].values

    X_test = features_df.loc[test_indices].values
    y_test = classes_df.loc[test_indices, 'label'].values

    # Split train into train and validation (80/20)
    n_train = int(0.8 * len(X_train_full))
    X_train = X_train_full[:n_train]
    y_train = y_train_full[:n_train]
    X_val = X_train_full[n_train:]
    y_val = y_train_full[n_train:]

    print("\nLoaded Elliptic++ dataset")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Test samples: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost Classical Baseline")
    parser.add_argument("--data-dir", default="data/elliptic", help="Elliptic dataset directory")
    parser.add_argument(
        "--model-output",
        default="experiments/shared_models/elliptic_xgboost_model.pkl",
        help="Path to save reusable XGBoost model",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--skip-save-model", action="store_true", help="Do not save trained model file")
    return parser.parse_args()


def main():
    """
    Main function to run XGBoost baseline experiment.
    """
    args = parse_args()

    print(f"\n{'#'*60}")
    print(f"# XGBoost Baseline Experiment")
    print(f"# Q-GAD: Quantum Graph Anomaly Detector")
    print(f"#" + "="*58 + "#")
    print(f"# Experiment: Pure classical features (166-dim)")
    print(f"# Model: XGBoost")
    print(f"# Purpose: Baseline without graph or quantum features")
    print(f"{'#'*60}\n")

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    data_dir = resolve_project_path(args.data_dir)
    X_train, y_train, X_val, y_val, X_test, y_test = load_elliptic_data(data_dir=str(data_dir))

    # Initialize model
    model = ClassicalOnlyBaseline(random_state=args.seed)

    # Train model
    model.fit(X_train, y_train, X_val, y_val, verbose=True)

    # Evaluate on test set
    metrics = model.evaluate(X_test, y_test, verbose=True)

    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "xgboost_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "experiment_name": "XGBoost Classical Baseline",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model_type": "XGBoost" if HAS_XGBOOST else "GradientBoosting",
        "feature_dim": 166,
        "uses_graph_topology": False,
        "uses_quantum_features": False,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "test_metrics": {
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "auc": float(metrics["auc"]),
            "ap": float(metrics["ap"]),
            "confusion_matrix": metrics["confusion_matrix"]
        }
    }

    output_file = output_dir / "xgboost_baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    model_output_path = resolve_project_path(args.model_output)
    if not args.skip_save_model:
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_output_path, "wb") as f:
            pickle.dump(model.model, f)
        print(f"Reusable model saved to: {model_output_path}")
        try:
            results["model_artifact"] = str(model_output_path.relative_to(PROJECT_ROOT))
        except ValueError:
            results["model_artifact"] = str(model_output_path)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*60}\n")

    # Print comparison summary
    print(f"\n{'='*60}")
    print(f"Experiment Summary")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  - Features: 166-dim classical features ONLY")
    print(f"  - Graph topology: NOT used")
    print(f"  - Quantum features: NOT used")
    print(f"  - Model: {results['model_type']}")
    print(f"\nPerformance:")
    print(f"  - F1 Score: {metrics['f1']:.4f}")
    print(f"  - AUC: {metrics['auc']:.4f}")
    print(f"  - Recall: {metrics['recall']:.4f}")
    print(f"\nNote:")
    print(f"  This baseline serves as the lower bound for comparison.")
    print(f"  Expected improvements when adding:")
    print(f"    1. Graph topology (GNN baselines)")
    print(f"    2. Quantum features (Q-GAD system)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
