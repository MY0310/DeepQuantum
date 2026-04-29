"""
Complete ablation training script.

Ablation variants:
1) Classical-only: 166D classical features
2) Quantum-only: precomputed quantum features
3) Hybrid: quantum + classical fusion

Key improvements:
- One-time quantum feature materialization (large speed-up)
- Validation-threshold search for best F1
- Class imbalance handling (weighted loss + optional balanced sampler)
- Early stopping and best-checkpoint restore
- Windows DataLoader worker fallback (WinError 5 -> num_workers=0)
"""

import argparse
import contextlib
import io
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.financial_dataset import collate_fn, load_elliptic_dataset
from gbs.gbs_kernel import GBSConfig, QuantumFeatureExtractor
from utils.helpers import find_best_f1_threshold, get_device, set_seed


# ============================================================================
# Model Definitions
# ============================================================================


class ClassicalOnlyModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64, 32), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, int(hidden_dim)),
                    nn.BatchNorm1d(int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, classical_features: torch.Tensor) -> torch.Tensor:
        return self.model(classical_features)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class QuantumOnlyModel(nn.Module):
    def __init__(self, quantum_dim: int, hidden_dims=(32, 16), dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = int(quantum_dim)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, int(hidden_dim)),
                    nn.BatchNorm1d(int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, 2))
        self.model = nn.Sequential(*layers)

    def forward(self, quantum_features: torch.Tensor) -> torch.Tensor:
        return self.model(quantum_features)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class HybridModel(nn.Module):
    def __init__(self, quantum_dim: int, classical_dim: int, dropout: float = 0.1):
        super().__init__()
        self.quantum_encoder = nn.Sequential(
            nn.Linear(int(quantum_dim), 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.classical_encoder = nn.Sequential(
            nn.Linear(int(classical_dim), 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(48, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(24, 2)

    def forward(self, quantum_features: torch.Tensor, classical_features: torch.Tensor) -> torch.Tensor:
        quantum_emb = self.quantum_encoder(quantum_features)
        classical_emb = self.classical_encoder(classical_features)
        combined = torch.cat([quantum_emb, classical_emb], dim=1)
        fused = self.fusion(combined)
        return self.classifier(fused)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# Data Containers
# ============================================================================


@dataclass
class MaterializedSplit:
    classical: torch.Tensor
    labels: torch.Tensor
    quantum: Optional[torch.Tensor] = None


class MaterializedDataset(Dataset):
    def __init__(self, split: MaterializedSplit):
        self.classical = split.classical
        self.labels = split.labels
        self.quantum = split.quantum

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "classical_features": self.classical[idx],
            "label": self.labels[idx],
        }
        if self.quantum is not None:
            item["quantum_features"] = self.quantum[idx]
        return item


# ============================================================================
# Utility Functions
# ============================================================================


def _extract_quantum_features(
    quantum_extractor: QuantumFeatureExtractor,
    squeezing: torch.Tensor,
    unitary: torch.Tensor,
    quiet_quantum_logs: bool = True,
) -> torch.Tensor:
    if not quiet_quantum_logs:
        return quantum_extractor(squeezing, unitary)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return quantum_extractor(squeezing, unitary)


def create_quantum_extractor(device: str, n_shots: int = 15) -> QuantumFeatureExtractor:
    gbs_config = GBSConfig(
        n_modes=20,
        n_shots=n_shots,
        backend="gaussian",
        use_displacement=True,
        device=device,
    )
    extractor = QuantumFeatureExtractor(gbs_config).to(device).eval()
    has_real = bool(getattr(extractor.gbs_kernel, "has_deepquantum", False))
    if not has_real:
        raise RuntimeError("DeepQuantum backend unavailable. Mock backend is not allowed.")
    return extractor


def _is_winerror5(exc: Exception) -> bool:
    text = str(exc).lower()
    return "winerror 5" in text or "access is denied" in text


def build_loader_with_fallback(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    collate=None,
    sampler=None,
    pin_memory: bool = False,
) -> Tuple[DataLoader, int]:
    effective_workers = max(0, int(num_workers))
    def _make_loader(n_workers: int) -> DataLoader:
        kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle if sampler is None else False,
            "num_workers": n_workers,
            "pin_memory": pin_memory,
            "sampler": sampler,
        }
        if collate is not None:
            kwargs["collate_fn"] = collate
        if n_workers > 0:
            kwargs["persistent_workers"] = True
        return DataLoader(**kwargs)

    try:
        loader = _make_loader(effective_workers)
    except Exception as exc:
        if effective_workers > 0 and _is_winerror5(exc):
            print("[Warn] DataLoader init failed with WinError 5. Falling back to num_workers=0.")
            return _make_loader(0), 0
        raise

    return loader, effective_workers


def stratified_train_val_split(
    dataset,
    val_ratio: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    labels = np.asarray(dataset.labels).astype(int)
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )
    return Subset(dataset, train_idx.tolist()), Subset(dataset, val_idx.tolist())


def maybe_subsample_subset(subset: Subset, ratio: float, seed: int) -> Subset:
    if ratio >= 1.0:
        return subset
    size = max(1, int(len(subset) * ratio))
    g = torch.Generator().manual_seed(seed)
    sampled, _ = torch.utils.data.random_split(subset, [size, len(subset) - size], generator=g)
    return sampled


def maybe_limit_subset(subset: Dataset, max_samples: Optional[int], seed: int) -> Dataset:
    if max_samples is None:
        return subset
    max_samples = int(max_samples)
    if len(subset) <= max_samples:
        return subset
    g = torch.Generator().manual_seed(seed)
    sampled, _ = torch.utils.data.random_split(subset, [max_samples, len(subset) - max_samples], generator=g)
    return sampled


def materialize_split(
    subset: Dataset,
    batch_size: int,
    device: str,
    num_workers: int,
    with_quantum: bool,
    quantum_extractor: Optional[QuantumFeatureExtractor] = None,
    quiet_quantum_logs: bool = True,
) -> Tuple[MaterializedSplit, int]:
    def _run(loader: DataLoader) -> MaterializedSplit:
        classical_list, labels_list = [], []
        quantum_list = [] if with_quantum else None

        pbar = tqdm(loader, desc="  Materializing split", ncols=100, leave=False)
        for batch in pbar:
            classical = batch["classical_features"].float().cpu()
            labels = batch["label"].long().cpu()
            classical_list.append(classical)
            labels_list.append(labels)

            if with_quantum:
                if quantum_extractor is None:
                    raise RuntimeError("quantum_extractor is required when with_quantum=True")
                squeezing = batch["squeezing"].to(device, non_blocking=(device == "cuda"))
                unitary = batch["unitary"].to(device, non_blocking=(device == "cuda"))
                with torch.no_grad():
                    qf = _extract_quantum_features(
                        quantum_extractor,
                        squeezing,
                        unitary,
                        quiet_quantum_logs=quiet_quantum_logs,
                    )
                quantum_list.append(qf.detach().float().cpu())

        return MaterializedSplit(
            classical=torch.cat(classical_list, dim=0),
            labels=torch.cat(labels_list, dim=0),
            quantum=(torch.cat(quantum_list, dim=0) if with_quantum else None),
        )

    loader, used_workers = build_loader_with_fallback(
        dataset=subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate=collate_fn,
        pin_memory=(device == "cuda"),
    )
    try:
        return _run(loader), used_workers
    except Exception as exc:
        if used_workers > 0 and _is_winerror5(exc):
            print("[Warn] DataLoader iteration failed with WinError 5. Retrying with num_workers=0.")
            loader0, _ = build_loader_with_fallback(
                dataset=subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate=collate_fn,
                pin_memory=(device == "cuda"),
            )
            return _run(loader0), 0
        raise


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= float(threshold)).astype(int)
    metrics = {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
    }
    metrics["auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    return metrics


def forward_model(model: nn.Module, batch: Dict[str, torch.Tensor], model_name: str, device: str) -> torch.Tensor:
    if model_name == "Classical":
        classical = batch["classical_features"].to(device, non_blocking=(device == "cuda"))
        return model(classical)
    if model_name == "Quantum":
        quantum = batch["quantum_features"].to(device, non_blocking=(device == "cuda"))
        return model(quantum)
    quantum = batch["quantum_features"].to(device, non_blocking=(device == "cuda"))
    classical = batch["classical_features"].to(device, non_blocking=(device == "cuda"))
    return model(quantum, classical)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    model_name: str,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"  Training {model_name}", ncols=100, leave=False)
    for batch in pbar:
        labels = batch["label"].to(device, non_blocking=(device == "cuda"))
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            outputs = forward_model(model, batch, model_name, device)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += float(loss.item())
        preds = torch.argmax(outputs.detach(), dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.shape[0])

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct / max(total, 1)):.4f}"})

    return total_loss / max(len(dataloader), 1), correct / max(total, 1)


def predict_probs(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    model_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device, non_blocking=(device == "cuda"))
            outputs = forward_model(model, batch, model_name, device)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_prob = torch.cat(all_probs).numpy()
    return y_true, y_prob


def get_class_weights(train_dataset: MaterializedDataset, device: str) -> torch.Tensor:
    labels = train_dataset.labels.numpy().astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    total = max(1, n_pos + n_neg)
    # Inverse-frequency style weights
    w_neg = total / max(1, 2 * n_neg)
    w_pos = total / max(1, 2 * n_pos)
    return torch.tensor([w_neg, w_pos], dtype=torch.float32, device=device)


def get_balanced_sampler(train_dataset: MaterializedDataset) -> WeightedRandomSampler:
    labels = train_dataset.labels.numpy().astype(int)
    class_counts = np.bincount(labels, minlength=2).astype(np.float64)
    inv = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = inv[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    model_name: str,
    output_dir: Path,
    epochs: int,
    lr: float,
    patience: int,
    decision_threshold: float,
    optimize_threshold: bool,
    use_class_weights: bool,
    train_dataset_for_weights: MaterializedDataset,
    use_amp: bool,
) -> Tuple[nn.Module, Dict[str, float]]:
    print(f"\n{'=' * 72}")
    print(f"Training {model_name}")
    print(f"{'=' * 72}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Epochs: {epochs}, LR: {lr}, Device: {device}")
    print(f"Optimize threshold: {optimize_threshold}, Base threshold: {decision_threshold:.3f}")
    print(f"Use class weights: {use_class_weights}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_class_weights:
        class_weights = get_class_weights(train_dataset_for_weights, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"Class weights [neg, pos]: {[float(class_weights[0]), float(class_weights[1])]}")
    else:
        criterion = nn.CrossEntropyLoss()

    best = {
        "val_f1": -1.0,
        "val_auc": -1.0,
        "epoch": 0,
        "threshold": float(decision_threshold),
    }
    bad_epochs = 0
    ckpt_path = output_dir / f"{model_name.lower()}_best.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            model_name=model_name,
            use_amp=use_amp,
        )

        y_val, p_val = predict_probs(model, val_loader, device, model_name)

        if optimize_threshold:
            threshold_obj = find_best_f1_threshold(y_true=y_val, y_prob=p_val)
            val_threshold = float(threshold_obj["threshold"])
            val_metrics = compute_binary_metrics(y_true=y_val, y_prob=p_val, threshold=val_threshold)
        else:
            val_threshold = float(decision_threshold)
            val_metrics = compute_binary_metrics(y_true=y_val, y_prob=p_val, threshold=val_threshold)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} | "
            f"ValF1={val_metrics['f1']:.4f} ValAUC={val_metrics['auc']:.4f} "
            f"ValThr={val_threshold:.3f}"
        )

        improved = val_metrics["f1"] > best["val_f1"] + 1e-6
        if improved:
            best["val_f1"] = float(val_metrics["f1"])
            best["val_auc"] = float(val_metrics["auc"])
            best["epoch"] = int(epoch)
            best["threshold"] = float(val_threshold)
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best["val_f1"],
                    "best_val_auc": best["val_auc"],
                    "best_threshold": best["threshold"],
                },
                ckpt_path,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"[EarlyStop] no ValF1 improvement for {patience} epochs. Stop at epoch {epoch}.")
            break

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best["threshold"] = float(ckpt.get("best_threshold", best["threshold"]))
    else:
        print("[Warn] best checkpoint missing; keeping last-epoch weights.")

    print(
        f"[Done] {model_name} best: ValF1={best['val_f1']:.4f}, "
        f"ValAUC={best['val_auc']:.4f}, Epoch={best['epoch']}, Thr={best['threshold']:.3f}"
    )
    return model, best


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str,
    model_name: str,
    threshold: float,
) -> Dict[str, float]:
    y_test, p_test = predict_probs(model, test_loader, device, model_name)
    metrics_best = compute_binary_metrics(y_true=y_test, y_prob=p_test, threshold=threshold)
    metrics_t05 = compute_binary_metrics(y_true=y_test, y_prob=p_test, threshold=0.5)
    out = {
        "f1": float(metrics_best["f1"]),
        "auc": float(metrics_best["auc"]),
        "recall": float(metrics_best["recall"]),
        "precision": float(metrics_best["precision"]),
        "threshold": float(threshold),
        "f1_t05": float(metrics_t05["f1"]),
        "recall_t05": float(metrics_t05["recall"]),
        "precision_t05": float(metrics_t05["precision"]),
    }
    return out


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Ablation study training")
    parser.add_argument("--model", type=str, default="all", choices=["all", "classical", "quantum", "hybrid"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--n-shots", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--optimize-threshold", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--balance-sampler", action="store_true")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-quantum-features", action="store_true")
    parser.add_argument("--show-quantum-logs", action="store_true")
    args = parser.parse_args()
    if os.name == "nt" and args.num_workers > 0:
        print("[Info] Windows session detected; forcing num_workers=0 to avoid WinError 5.")
        args.num_workers = 0

    set_seed(args.seed)
    device = get_device(args.device)
    use_amp = device == "cuda"

    print("\n" + "=" * 72)
    print("Q-GAD Ablation Training")
    print("=" * 72)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"Subset: {args.subset}, n_shots: {args.n_shots}, num_workers(req): {args.num_workers}")
    print(f"Optimize threshold: {args.optimize_threshold}, class weights: {not args.no_class_weights}")
    print(f"Balance sampler: {args.balance_sampler}, AMP: {use_amp}")
    print("=" * 72)

    # 1) Load dataset
    print("\n[1/6] Loading Elliptic dataset...")
    train_dataset, test_dataset = load_elliptic_dataset(
        data_dir="data/elliptic",
        max_nodes=20,
        ego_radius=1.5,
        cache_dir="data/elliptic/processed",
    )

    # 2) Stratified split and optional subsampling
    print("\n[2/6] Building stratified train/val split...")
    train_subset, val_subset = stratified_train_val_split(
        dataset=train_dataset,
        val_ratio=0.2,
        seed=args.seed,
    )

    train_subset = maybe_subsample_subset(train_subset, args.subset, args.seed)
    train_subset = maybe_limit_subset(train_subset, args.max_train_samples, args.seed)
    val_subset = maybe_limit_subset(val_subset, args.max_val_samples, args.seed)
    test_subset = maybe_limit_subset(test_dataset, args.max_test_samples, args.seed)

    print(f"Train subset: {len(train_subset)}")
    print(f"Val subset:   {len(val_subset)}")
    print(f"Test subset:  {len(test_subset)}")

    # 3) Materialize split (and quantum once if needed)
    quiet_quantum_logs = not args.show_quantum_logs
    need_quantum = args.model in ("all", "quantum", "hybrid")
    cache_dir = Path(__file__).parent / "cache"
    cache_file = cache_dir / (
        f"materialized_seed{args.seed}_shots{args.n_shots}_"
        f"tr{len(train_subset)}_va{len(val_subset)}_te{len(test_subset)}.pt"
    )

    print("\n[3/6] Materializing features...")
    loaded_from_cache = False
    if args.cache_quantum_features and cache_file.exists():
        try:
            payload = torch.load(cache_file, map_location="cpu", weights_only=False)
            train_split = payload["train"]
            val_split = payload["val"]
            test_split = payload["test"]
            loaded_from_cache = True
            effective_workers = 0
            print(f"[OK] Reused materialized cache: {cache_file}")
        except Exception as exc:
            print(f"[Warn] Failed to load cache ({cache_file}): {exc}. Recomputing...")
            loaded_from_cache = False

    if not loaded_from_cache:
        quantum_extractor = None
        if need_quantum:
            print("Initializing real DeepQuantum extractor...")
            quantum_extractor = create_quantum_extractor(device=device, n_shots=args.n_shots)
            print("[OK] DeepQuantum extractor ready.")

        train_split, workers_used = materialize_split(
            subset=train_subset,
            batch_size=args.batch_size,
            device=device,
            num_workers=args.num_workers,
            with_quantum=need_quantum,
            quantum_extractor=quantum_extractor,
            quiet_quantum_logs=quiet_quantum_logs,
        )
        val_split, workers_used_val = materialize_split(
            subset=val_subset,
            batch_size=args.batch_size,
            device=device,
            num_workers=workers_used,
            with_quantum=need_quantum,
            quantum_extractor=quantum_extractor,
            quiet_quantum_logs=quiet_quantum_logs,
        )
        test_split, workers_used_test = materialize_split(
            subset=test_subset,
            batch_size=args.batch_size,
            device=device,
            num_workers=min(workers_used, workers_used_val),
            with_quantum=need_quantum,
            quantum_extractor=quantum_extractor,
            quiet_quantum_logs=quiet_quantum_logs,
        )
        effective_workers = min(workers_used, workers_used_val, workers_used_test)
        print(f"Effective DataLoader workers: {effective_workers}")

        # Optional disk cache for materialized quantum features
        if args.cache_quantum_features:
            cache_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "train": train_split,
                    "val": val_split,
                    "test": test_split,
                },
                cache_file,
            )
            print(f"[OK] Materialized cache saved: {cache_file}")

        # Free extractor after materialization
        if quantum_extractor is not None:
            del quantum_extractor
            if device == "cuda":
                torch.cuda.empty_cache()

    classical_dim = int(train_split.classical.shape[1])
    quantum_dim = int(train_split.quantum.shape[1]) if need_quantum and train_split.quantum is not None else 0

    train_ds = MaterializedDataset(train_split)
    val_ds = MaterializedDataset(val_split)
    test_ds = MaterializedDataset(test_split)

    # 4) Build training/eval loaders
    print("\n[4/6] Building DataLoaders for model training...")
    sampler = get_balanced_sampler(train_ds) if args.balance_sampler else None
    train_loader, _ = build_loader_with_fallback(
        dataset=train_ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=effective_workers,
        collate=None,
        sampler=sampler,
        pin_memory=(device == "cuda"),
    )
    val_loader, _ = build_loader_with_fallback(
        dataset=val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_workers,
        collate=None,
        pin_memory=(device == "cuda"),
    )
    test_loader, _ = build_loader_with_fallback(
        dataset=test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_workers,
        collate=None,
        pin_memory=(device == "cuda"),
    )

    # 5) Train selected models
    print("\n[5/6] Training selected ablation variants...")
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(__file__).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}

    def _run_single(name: str, model: nn.Module):
        trained_model, best = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            model_name=name,
            output_dir=ckpt_dir,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            decision_threshold=args.decision_threshold,
            optimize_threshold=args.optimize_threshold,
            use_class_weights=(not args.no_class_weights),
            train_dataset_for_weights=train_ds,
            use_amp=use_amp,
        )
        test_metrics = evaluate_model(
            model=trained_model,
            test_loader=test_loader,
            device=device,
            model_name=name,
            threshold=best["threshold"],
        )
        test_metrics["best_val_f1"] = float(best["val_f1"])
        test_metrics["best_val_auc"] = float(best["val_auc"])
        test_metrics["best_epoch"] = int(best["epoch"])
        results[name] = test_metrics
        print(
            f"Test {name}: F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}, "
            f"Recall={test_metrics['recall']:.4f}, Prec={test_metrics['precision']:.4f}, "
            f"Thr={test_metrics['threshold']:.3f}"
        )

    if args.model in ("all", "classical"):
        _run_single("Classical", ClassicalOnlyModel(input_dim=classical_dim))
    if args.model in ("all", "quantum"):
        _run_single("Quantum", QuantumOnlyModel(quantum_dim=quantum_dim))
    if args.model in ("all", "hybrid"):
        _run_single("Hybrid", HybridModel(quantum_dim=quantum_dim, classical_dim=classical_dim))

    # 6) Save outputs
    print("\n[6/6] Saving outputs...")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = str(args.model).lower()
    json_path = output_dir / f"ablation_training_{model_tag}_{ts}.json"
    csv_path = output_dir / f"ablation_training_{model_tag}_{ts}.csv"

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "subset": float(args.subset),
        "seed": int(args.seed),
        "n_shots": int(args.n_shots),
        "decision_threshold": float(args.decision_threshold),
        "optimize_threshold": bool(args.optimize_threshold),
        "balance_sampler": bool(args.balance_sampler),
        "class_weights": bool(not args.no_class_weights),
        "patience": int(args.patience),
        "num_workers_effective": int(effective_workers),
        "results": results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    rows = []
    for name, m in results.items():
        rows.append(
            {
                "Model": name,
                "F1": m["f1"],
                "AUC": m["auc"],
                "Recall": m["recall"],
                "Precision": m["precision"],
                "Threshold": m["threshold"],
                "F1@0.5": m["f1_t05"],
                "Recall@0.5": m["recall_t05"],
                "Precision@0.5": m["precision_t05"],
                "BestValF1": m["best_val_f1"],
                "BestValAUC": m["best_val_auc"],
                "BestEpoch": m["best_epoch"],
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"[OK] JSON saved: {json_path}")
    print(f"[OK] CSV saved:  {csv_path}")

    print("\n" + "=" * 72)
    print("Ablation Summary")
    print("=" * 72)
    print(f"{'Model':<12} {'F1':<8} {'AUC':<8} {'Recall':<8} {'Prec':<8} {'Thr':<6}")
    print("-" * 72)
    for name, m in results.items():
        print(
            f"{name:<12} {m['f1']:<8.4f} {m['auc']:<8.4f} {m['recall']:<8.4f} "
            f"{m['precision']:<8.4f} {m['threshold']:<6.3f}"
        )
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
