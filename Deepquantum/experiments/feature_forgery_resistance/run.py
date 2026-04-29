"""
Unified feature forgery resistance experiment.

Goals:
1. Use real Q-GAD model (no mock DeepQuantum backend allowed).
2. Use real classical baseline (cached XGBoost if available).
3. Run fast on a subset of cached Elliptic++ samples.
4. Compare fraud detection retention after feature forgery attacks.

Usage:
    python experiments/feature_forgery_resistance/run.py --n-samples 64 --optimization-steps 3 --device cpu
"""

import argparse
import contextlib
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, Dataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.financial_dataset import inspect_quantum_cache, is_quantum_cache_usable
from utils.elliptic_xgb import load_or_train_elliptic_xgb
from utils.helpers import (
    get_device,
    load_qgad_checkpoint_model,
    logits_to_binary_predictions,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified feature forgery resistance experiment")
    parser.add_argument("--n-samples", type=int, default=64, help="Number of fraud samples to evaluate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--optimization-steps", type=int, default=3, help="Attack optimization steps")
    parser.add_argument("--forgery-budget", type=float, default=0.1, help="L_inf perturbation budget")
    parser.add_argument("--n-shots", type=int, default=20, help="DeepQuantum shots per sample (speed/variance tradeoff)")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Fraud decision threshold on P(y=1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default=None, help="cuda/cpu/mps, default auto")
    parser.add_argument("--show-quantum-logs", action="store_true", help="Show verbose DeepQuantum per-sample logs")
    parser.add_argument(
        "--xgb-model-path",
        default="experiments/shared_models/elliptic_xgboost_model.pkl",
        help="Shared XGBoost model path for reuse across experiments",
    )
    parser.add_argument("--retrain-xgb", action="store_true", help="Force retrain XGBoost baseline model")
    parser.add_argument(
        "--output-dir",
        default="experiments/feature_forgery_resistance/results",
        help="Directory for outputs",
    )
    return parser.parse_args()


def _normalize_labels(classes: pd.Series) -> pd.Series:
    """
    Normalize raw class labels to:
    - 1: fraud
    - 0: licit
    - -1: unknown
    """
    if classes.dtype == object:
        mapped = classes.map({"unknown": -1, "1": 1, "2": 0}).where(classes.isin(["unknown", "1", "2"]), classes)
        numeric = pd.to_numeric(mapped, errors="coerce")
    else:
        numeric = pd.to_numeric(classes, errors="coerce")

    numeric = numeric.fillna(-1).astype(int)
    uniq = set(numeric.unique().tolist())

    # Handle raw 0/1/2 style from Elliptic docs: 0=unknown, 1=illicit, 2=licit.
    if 2 in uniq:
        numeric = numeric.map({0: -1, 1: 1, 2: 0}).fillna(-1).astype(int)

    return numeric


class CachedFraudSubset(Dataset):
    """
    Fast dataset using cached quantum parameters + raw features/classes.
    """

    def __init__(
        self,
        n_samples: int,
        seed: int = 42,
        fraud_only: bool = True,
        cache_path: str = "data/elliptic/processed/quantum_params_m20_r1.5.pkl",
        features_path: str = "data/elliptic/raw/features.csv",
        classes_path: str = "data/elliptic/raw/classes.csv",
    ):
        self.seed = seed
        self.fraud_only = fraud_only

        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if not is_quantum_cache_usable(cache, max_nodes=20):
            report = inspect_quantum_cache(cache, max_nodes=20)
            raise RuntimeError(
                "Invalid/degenerate quantum cache detected. "
                f"sq_zero_ratio={report.get('sq_zero_ratio', -1):.4f}, "
                f"unitary_identity_ratio={report.get('unitary_identity_ratio', -1):.4f}, "
                f"center_ratio={report.get('center_ratio', -1):.4f}. "
                "Please regenerate cache by re-running dataset preprocessing/training scripts."
            )

        required = {"squeezing", "unitary", "metadata"}
        missing = required - set(cache.keys())
        if missing:
            raise ValueError(f"Cache file missing keys: {sorted(missing)}")

        self.squeezing = cache["squeezing"]
        self.unitary = cache["unitary"]
        metadata = cache["metadata"]

        node_to_cache_idx: Dict[int, int] = {}
        for idx, meta in enumerate(metadata):
            center = meta.get("center_node") if isinstance(meta, dict) else None
            if center is None:
                continue
            center = int(center)
            if center not in node_to_cache_idx:
                node_to_cache_idx[center] = idx

        if not node_to_cache_idx:
            raise RuntimeError("No center_node mapping found in cache metadata.")

        features_df = pd.read_csv(features_path, index_col=0)
        classes_df = pd.read_csv(classes_path, index_col=0)
        labels = _normalize_labels(classes_df["class"])

        candidate_nodes = set(node_to_cache_idx.keys()) & set(features_df.index.tolist()) & set(labels.index.tolist())
        if fraud_only:
            fraud_nodes = set(labels[labels == 1].index.tolist())
            candidate_nodes &= fraud_nodes
        else:
            known_nodes = set(labels[labels >= 0].index.tolist())
            candidate_nodes &= known_nodes

        if not candidate_nodes:
            raise RuntimeError("No matching labeled samples found between cache/features/classes.")

        nodes_sorted = np.array(sorted(candidate_nodes), dtype=np.int64)
        rng = np.random.default_rng(seed)
        take = min(n_samples, len(nodes_sorted))
        selected = rng.choice(nodes_sorted, size=take, replace=False)
        selected = sorted(selected.tolist())

        self.cache_indices: List[int] = [node_to_cache_idx[int(n)] for n in selected]
        self.features = features_df.loc[selected].to_numpy(dtype=np.float32)
        self.labels = labels.loc[selected].to_numpy(dtype=np.int64)
        self.nodes = selected

        fraud_cnt = int((self.labels == 1).sum())
        licit_cnt = int((self.labels == 0).sum())
        print(f"[Dataset] Loaded {len(self.nodes)} samples (fraud={fraud_cnt}, licit={licit_cnt})")

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        cidx = self.cache_indices[idx]
        return {
            "squeezing": torch.tensor(self.squeezing[cidx], dtype=torch.float32),
            "unitary": torch.tensor(self.unitary[cidx], dtype=torch.float32),
            "classical_features": torch.tensor(self.features[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_qgad_model(device: str, n_shots: int):
    checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "elliptic_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model, checkpoint, inferred = load_qgad_checkpoint_model(
        checkpoint_path=str(checkpoint_path),
        device=device,
        n_modes=20,
        n_shots=n_shots,
    )

    has_real_backend = bool(getattr(model.quantum_extractor.gbs_kernel, "has_deepquantum", False))
    best_f1 = max(checkpoint.get("history", {}).get("val_f1", [0.0]))
    print(f"[Q-GAD] Loaded checkpoint {checkpoint_path.name} (best val_f1={best_f1:.4f}, n_shots={n_shots})")
    print(f"[Q-GAD] Inferred architecture hidden_dims={inferred['hidden_dims']}")
    print(f"[Q-GAD] Real DeepQuantum backend: {has_real_backend}")
    if not has_real_backend:
        raise RuntimeError(
            "DeepQuantum backend not available. This experiment forbids mock backend. "
            "Please run in an environment with deepquantum installed."
        )

    return model


def _qgad_forward(
    model,
    squeezing: torch.Tensor,
    unitary: torch.Tensor,
    classical: torch.Tensor,
    quiet_quantum_logs: bool = True,
):
    if not quiet_quantum_logs:
        return model(squeezing, unitary, classical)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return model(squeezing, unitary, classical)


@torch.no_grad()
def evaluate_qgad(
    model,
    loader: DataLoader,
    device: str,
    threshold: float = 0.5,
    quiet_quantum_logs: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    all_preds, all_labels = [], []
    for batch in loader:
        logits = _qgad_forward(
            model,
            batch["squeezing"],
            batch["unitary"],
            batch["classical_features"].to(device),
            quiet_quantum_logs=quiet_quantum_logs,
        )
        _, preds = logits_to_binary_predictions(logits, threshold=threshold)
        preds = preds.cpu().numpy()
        labels = batch["label"].cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return np.array(all_preds, dtype=np.int64), np.array(all_labels, dtype=np.int64)


def attack_qgad_batch(
    model,
    squeezing: torch.Tensor,
    unitary: torch.Tensor,
    classical: torch.Tensor,
    steps: int,
    budget: float,
    quiet_quantum_logs: bool = True,
) -> torch.Tensor:
    target = torch.zeros(classical.shape[0], dtype=torch.long, device=classical.device)
    forged = classical.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([forged], lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    for _ in range(steps):
        optimizer.zero_grad()
        logits = _qgad_forward(
            model, squeezing, unitary, forged, quiet_quantum_logs=quiet_quantum_logs
        )
        loss = loss_fn(logits, target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta = torch.clamp(forged - classical, -budget, budget)
            forged.copy_(classical + delta)

    return forged.detach()


def evaluate_qgad_under_attack(
    model,
    loader: DataLoader,
    device: str,
    steps: int,
    budget: float,
    threshold: float = 0.5,
    quiet_quantum_logs: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    all_preds, all_labels = [], []
    for batch in loader:
        squeezing = batch["squeezing"]
        unitary = batch["unitary"]
        classical = batch["classical_features"].to(device)
        labels = batch["label"].cpu().numpy()

        forged = attack_qgad_batch(
            model,
            squeezing,
            unitary,
            classical,
            steps=steps,
            budget=budget,
            quiet_quantum_logs=quiet_quantum_logs,
        )
        with torch.no_grad():
            logits = _qgad_forward(
                model, squeezing, unitary, forged, quiet_quantum_logs=quiet_quantum_logs
            )
            _, preds = logits_to_binary_predictions(logits, threshold=threshold)
            preds = preds.cpu().numpy()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return np.array(all_preds, dtype=np.int64), np.array(all_labels, dtype=np.int64)


def evaluate_xgb(model, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    all_preds, all_labels = [], []
    for batch in loader:
        x = batch["classical_features"].numpy()
        y = batch["label"].numpy()
        preds = model.predict(x)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())
    return np.array(all_preds, dtype=np.int64), np.array(all_labels, dtype=np.int64)


def attack_xgb_batch(model, x: np.ndarray, steps: int, budget: float, seed: int = 42) -> np.ndarray:
    """
    Gradient-free targeted attack: search perturbations to reduce fraud probability.
    """
    rng = np.random.default_rng(seed)
    base = x.astype(np.float32, copy=True)
    low = base - budget
    high = base + budget

    best = base.copy()
    best_score = model.predict_proba(best)[:, 1]  # fraud probability

    for _ in range(steps):
        candidate = base + rng.uniform(-budget, budget, size=base.shape).astype(np.float32)
        candidate = np.clip(candidate, low, high)
        score = model.predict_proba(candidate)[:, 1]
        improved = score < best_score
        best_score[improved] = score[improved]
        best[improved] = candidate[improved]

    return best


def evaluate_xgb_under_attack(
    model,
    loader: DataLoader,
    steps: int,
    budget: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    all_preds, all_labels = [], []
    for idx, batch in enumerate(loader):
        x = batch["classical_features"].numpy()
        y = batch["label"].numpy()
        forged = attack_xgb_batch(model, x, steps=steps, budget=budget, seed=seed + idx)
        preds = model.predict(forged)
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())
    return np.array(all_preds, dtype=np.int64), np.array(all_labels, dtype=np.int64)


def fraud_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Fraud class is label=1
    return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))


def _build_loader(dataset, batch_size, num_workers):
    workers = max(0, int(num_workers))
    if workers <= 0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0), 0
    try:
        import multiprocessing as mp
        r, w = mp.Pipe(duplex=False)
        r.close()
        w.close()
    except Exception as e:
        print(f"[DataLoader] num_workers={workers} unavailable ({e}); fallback to 0.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0), 0

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers), workers


def run(args: argparse.Namespace) -> Dict:
    set_seed(args.seed)
    device = get_device(args.device)
    print(
        f"[Config] device={device}, n_samples={args.n_samples}, steps={args.optimization_steps}, "
        f"budget={args.forgery_budget}, threshold={args.decision_threshold:.3f}"
    )

    dataset = CachedFraudSubset(n_samples=args.n_samples, seed=args.seed, fraud_only=True)
    loader, active_workers = _build_loader(dataset, args.batch_size, args.num_workers)
    print(f"[DataLoader] num_workers={active_workers}")

    qgad_model = load_qgad_model(device=device, n_shots=args.n_shots)
    xgb_model, xgb_model_path, xgb_training_report = load_or_train_elliptic_xgb(
        model_path_like=args.xgb_model_path,
        seed=args.seed,
        force_retrain=args.retrain_xgb,
    )
    quiet_quantum_logs = not args.show_quantum_logs

    print("\n[1/4] Baseline evaluation...")
    qgad_base_pred, qgad_base_true = evaluate_qgad(
        qgad_model,
        loader,
        device=device,
        threshold=args.decision_threshold,
        quiet_quantum_logs=quiet_quantum_logs,
    )
    xgb_base_pred, xgb_base_true = evaluate_xgb(xgb_model, loader)

    print("[2/4] Q-GAD forgery attack...")
    qgad_adv_pred, qgad_adv_true = evaluate_qgad_under_attack(
        qgad_model,
        loader,
        device=device,
        steps=args.optimization_steps,
        budget=args.forgery_budget,
        threshold=args.decision_threshold,
        quiet_quantum_logs=quiet_quantum_logs,
    )

    print("[3/4] XGBoost forgery attack...")
    xgb_adv_pred, xgb_adv_true = evaluate_xgb_under_attack(
        xgb_model,
        loader,
        steps=args.optimization_steps,
        budget=args.forgery_budget,
        seed=args.seed,
    )

    qgad_base_recall = fraud_recall(qgad_base_true, qgad_base_pred)
    qgad_adv_recall = fraud_recall(qgad_adv_true, qgad_adv_pred)
    xgb_base_recall = fraud_recall(xgb_base_true, xgb_base_pred)
    xgb_adv_recall = fraud_recall(xgb_adv_true, xgb_adv_pred)

    qgad_drop = qgad_base_recall - qgad_adv_recall
    xgb_drop = xgb_base_recall - xgb_adv_recall
    drop_gap = xgb_drop - qgad_drop

    print("\n[4/4] Summary")
    print("-" * 72)
    print(f"{'Model':<16}{'Base Recall':<14}{'Forged Recall':<16}{'Drop':<10}")
    print(f"{'Q-GAD':<16}{qgad_base_recall:<14.4f}{qgad_adv_recall:<16.4f}{qgad_drop:<10.4f}")
    print(f"{'XGBoost':<16}{xgb_base_recall:<14.4f}{xgb_adv_recall:<16.4f}{xgb_drop:<10.4f}")
    print("-" * 72)
    print(f"Q-GAD drop advantage vs XGBoost: {drop_gap:.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "feature_forgery_resistance_unified_fast",
        "parameters": {
            "n_samples": int(args.n_samples),
            "batch_size": int(args.batch_size),
            "optimization_steps": int(args.optimization_steps),
            "forgery_budget": float(args.forgery_budget),
            "n_shots": int(args.n_shots),
            "decision_threshold": float(args.decision_threshold),
            "seed": int(args.seed),
            "device": str(device),
        },
        "qgad": {
            "baseline_recall": qgad_base_recall,
            "forged_recall": qgad_adv_recall,
            "recall_drop": qgad_drop,
        },
        "xgboost": {
            "baseline_recall": xgb_base_recall,
            "forged_recall": xgb_adv_recall,
            "recall_drop": xgb_drop,
        },
        "advantage": {
            "drop_gap_xgb_minus_qgad": drop_gap,
            "qgad_more_robust": bool(drop_gap > 0),
        },
        "backend": {
            "deepquantum_real": True,
            "checkpoint": "elliptic_model.pt",
        },
        "artifacts": {
            "xgboost_model_path": str(xgb_model_path),
        },
    }
    if xgb_training_report is not None:
        result["xgboost_training"] = xgb_training_report

    json_path = output_dir / "feature_forgery_unified_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {json_path}")
    return result


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
