"""
Adversarial robustness experiment (standardized, real model).
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.financial_dataset import collate_fn, load_elliptic_dataset
from utils.helpers import (
    get_device,
    load_qgad_checkpoint_model,
    logits_to_binary_predictions,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial Robustness (Standardized)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-shots", type=int, default=15)
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Fraud decision threshold on P(y=1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-quantum-logs", action="store_true")
    parser.add_argument(
        "--attack-feature-ratio",
        type=float,
        default=1.0,
        help="Fraction of classical features perturbed per sample (top-|grad|). 1.0 means all features.",
    )
    parser.add_argument(
        "--materialize-quantum",
        dest="materialize_quantum",
        action="store_true",
        help="Precompute quantum features once and reuse for all attacks (faster, equivalent).",
    )
    parser.add_argument(
        "--no-materialize-quantum",
        dest="materialize_quantum",
        action="store_false",
        help="Disable quantum feature materialization.",
    )
    parser.add_argument(
        "--clip-to-train-range",
        dest="clip_to_train_range",
        action="store_true",
        help="Project adversarial classical features to train-set feature range.",
    )
    parser.add_argument(
        "--no-clip-to-train-range",
        dest="clip_to_train_range",
        action="store_false",
        help="Disable projection to train-set feature range (legacy behavior).",
    )
    parser.add_argument(
        "--cache-materialized",
        dest="cache_materialized",
        action="store_true",
        help="Cache precomputed (quantum,classical,label) tensors for repeated runs.",
    )
    parser.add_argument(
        "--no-cache-materialized",
        dest="cache_materialized",
        action="store_false",
        help="Disable materialized tensor cache.",
    )
    parser.add_argument("--cache-dir", default="experiments/adversarial_robustness/cache")
    parser.add_argument("--output-dir", default="experiments/adversarial_robustness/results")
    parser.set_defaults(materialize_quantum=True, clip_to_train_range=True, cache_materialized=True)
    return parser.parse_args()


def _safe_auc(y_true, y_prob) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.0


def _forward(model, squeezing, unitary, classical, quiet_quantum_logs=True):
    if not quiet_quantum_logs:
        return model(squeezing, unitary, classical)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return model(squeezing, unitary, classical)


def _extract_quantum(model, squeezing, unitary, quiet_quantum_logs=True):
    if not quiet_quantum_logs:
        return model.quantum_extractor(squeezing, unitary)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return model.quantum_extractor(squeezing, unitary)


def _random_subset(dataset, subset_size: int, seed: int):
    if subset_size >= len(dataset):
        return dataset
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    indices = torch.randperm(len(dataset), generator=gen)[:subset_size].tolist()
    return Subset(dataset, indices)


def _stratified_subset(dataset, subset_size: int, seed: int):
    if subset_size >= len(dataset):
        return dataset
    labels = getattr(dataset, "labels", None)
    if labels is None:
        return _random_subset(dataset, subset_size, seed)
    labels = torch.as_tensor(labels).cpu().numpy()
    if labels.shape[0] != len(dataset):
        return _random_subset(dataset, subset_size, seed)

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return _random_subset(dataset, subset_size, seed)

    rng = np.random.default_rng(seed)
    half = subset_size // 2
    n_pos = min(len(pos_idx), half)
    n_neg = min(len(neg_idx), half)
    remain = subset_size - (n_pos + n_neg)
    if remain > 0:
        extra_pos = min(len(pos_idx) - n_pos, remain)
        n_pos += extra_pos
        remain -= extra_pos
    if remain > 0:
        extra_neg = min(len(neg_idx) - n_neg, remain)
        n_neg += extra_neg

    chosen_pos = rng.choice(pos_idx, size=n_pos, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    indices = np.concatenate([chosen_pos, chosen_neg])
    rng.shuffle(indices)
    return Subset(dataset, indices.tolist())


def _build_loader(dataset, batch_size, num_workers):
    workers = max(0, int(num_workers))
    if workers <= 0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0), 0

    try:
        import multiprocessing as mp
        r, w = mp.Pipe(duplex=False)
        r.close()
        w.close()
    except Exception as e:
        print(f"[DataLoader] num_workers={workers} unavailable ({e}); fallback to 0.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0), 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=workers,
    ), workers


class MaterializedAttackDataset(Dataset):
    def __init__(self, classical: torch.Tensor, quantum: torch.Tensor, labels: torch.Tensor):
        self.classical = classical
        self.quantum = quantum
        self.labels = labels

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return {
            "classical_features": self.classical[idx],
            "quantum_features": self.quantum[idx],
            "label": self.labels[idx],
        }


def _build_materialized_loader(dataset, batch_size, num_workers):
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


def _compute_feature_bounds(train_dataset, device):
    features = getattr(train_dataset, "node_features", None)
    if features is None:
        return None
    feat_t = torch.as_tensor(features, dtype=torch.float32, device=device)
    if feat_t.ndim != 2 or feat_t.shape[1] <= 0:
        return None
    feature_min = feat_t.min(dim=0).values
    feature_max = feat_t.max(dim=0).values
    return feature_min, feature_max


@torch.no_grad()
def materialize_quantum_batches(model, raw_loader, device, quiet_quantum_logs=True):
    classical_list, quantum_list, labels_list = [], [], []
    t0 = time.time()
    for batch in raw_loader:
        classical = batch["classical_features"].to(device)
        squeezing = batch["squeezing"]
        unitary = batch["unitary"]
        quantum = _extract_quantum(
            model=model,
            squeezing=squeezing,
            unitary=unitary,
            quiet_quantum_logs=quiet_quantum_logs,
        )
        classical_list.append(classical.cpu())
        quantum_list.append(quantum.detach().cpu())
        labels_list.append(batch["label"].cpu())
    classical = torch.cat(classical_list, dim=0)
    quantum = torch.cat(quantum_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    elapsed = time.time() - t0
    print(f"[Cache] Materialized quantum features for {labels.shape[0]} samples in {elapsed:.1f}s")
    return MaterializedAttackDataset(classical=classical, quantum=quantum, labels=labels)


def load_qgad_model(device: str, n_shots: int):
    checkpoint_path = Path(__file__).parent.parent.parent / "checkpoints" / "elliptic_model.pt"

    model, checkpoint, inferred = load_qgad_checkpoint_model(
        checkpoint_path=str(checkpoint_path),
        device=device,
        n_modes=20,
        n_shots=n_shots,
    )

    has_real_backend = bool(getattr(model.quantum_extractor.gbs_kernel, "has_deepquantum", False))
    best_f1 = max(checkpoint.get("history", {}).get("val_f1", [0.0]))
    print(f"[Model] Loaded {checkpoint_path.name} (best val_f1={best_f1:.4f}, n_shots={n_shots})")
    print(f"[Model] Inferred architecture hidden_dims={inferred['hidden_dims']}")
    print(f"[Model] Real DeepQuantum backend: {has_real_backend}")
    if not has_real_backend:
        raise RuntimeError("DeepQuantum backend unavailable. Mock backend is not allowed.")
    return model


@torch.no_grad()
def evaluate_clean(model, test_loader, device, threshold=0.5, quiet_quantum_logs=True):
    all_preds, all_labels, all_probs = [], [], []
    for batch in test_loader:
        classical = batch["classical_features"].to(device)
        if "quantum_features" in batch:
            quantum = batch["quantum_features"].to(device)
            logits = model.hybrid_classifier(classical, quantum)
        else:
            logits = _forward(
                model,
                batch["squeezing"],
                batch["unitary"],
                classical,
                quiet_quantum_logs=quiet_quantum_logs,
            )
        probs_pos, preds = logits_to_binary_predictions(logits, threshold=threshold)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"].cpu().numpy())
        all_probs.extend(probs_pos.cpu().numpy())
    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "auc": _safe_auc(all_labels, all_probs),
    }


def evaluate_under_attack_multi_eps(
    model,
    test_loader,
    epsilons,
    device,
    threshold=0.5,
    quiet_quantum_logs=True,
    feature_bounds=None,
    attack_feature_ratio=1.0,
):
    eps_list = [float(e) for e in epsilons]
    stores = {
        eps: {"preds": [], "labels": [], "probs": []}
        for eps in eps_list
    }
    if feature_bounds is not None:
        f_min, f_max = feature_bounds
        f_min = f_min.view(1, -1)
        f_max = f_max.view(1, -1)
    else:
        f_min, f_max = None, None

    for batch in test_loader:
        classical = batch["classical_features"].to(device)
        labels = batch["label"].to(device)
        quantum = batch.get("quantum_features", None)
        if quantum is not None:
            quantum = quantum.to(device)

        classical_adv = classical.clone().detach().requires_grad_(True)
        if quantum is not None:
            logits = model.hybrid_classifier(classical_adv, quantum)
        else:
            logits = _forward(
                model,
                batch["squeezing"],
                batch["unitary"],
                classical_adv,
                quiet_quantum_logs=quiet_quantum_logs,
            )
        loss = F.cross_entropy(logits, labels)
        model.zero_grad()
        loss.backward()
        grad = classical_adv.grad
        grad_sign = grad.sign()
        ratio = float(attack_feature_ratio)
        if ratio < 1.0:
            ratio = max(0.0, ratio)
            k = max(1, int(grad.shape[1] * ratio))
            top_idx = torch.topk(grad.abs(), k=k, dim=1).indices
            mask = torch.zeros_like(grad_sign)
            mask.scatter_(1, top_idx, 1.0)
            grad_sign = grad_sign * mask
        labels_np = labels.detach().cpu().numpy()

        for eps in eps_list:
            perturbation = eps * grad_sign
            classical_eps = (classical + perturbation).detach()
            if f_min is not None and f_max is not None:
                classical_eps = torch.max(torch.min(classical_eps, f_max), f_min)
            else:
                classical_eps = torch.clamp(classical_eps, -10, 10)

            with torch.no_grad():
                if quantum is not None:
                    logits_eps = model.hybrid_classifier(classical_eps, quantum)
                else:
                    logits_eps = _forward(
                        model,
                        batch["squeezing"],
                        batch["unitary"],
                        classical_eps,
                        quiet_quantum_logs=quiet_quantum_logs,
                    )
            probs_pos, preds = logits_to_binary_predictions(logits_eps, threshold=threshold)
            stores[eps]["preds"].extend(preds.cpu().numpy())
            stores[eps]["labels"].extend(labels_np)
            stores[eps]["probs"].extend(probs_pos.cpu().numpy())

    out = {}
    for eps in eps_list:
        out[eps] = {
            "accuracy": float(accuracy_score(stores[eps]["labels"], stores[eps]["preds"])),
            "f1": float(f1_score(stores[eps]["labels"], stores[eps]["preds"], zero_division=0)),
            "auc": _safe_auc(stores[eps]["labels"], stores[eps]["probs"]),
        }
    return out


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    quiet_quantum_logs = not args.show_quantum_logs

    print("\n" + "=" * 80)
    print("Adversarial Robustness (Standardized)")
    print("=" * 80)
    print(
        f"[Config] device={device}, quick={args.quick}, epsilon={args.epsilon}, "
        f"n_shots={args.n_shots}, threshold={args.decision_threshold:.3f}"
    )

    model = load_qgad_model(device=device, n_shots=args.n_shots)

    train_dataset, test_dataset = load_elliptic_dataset(
        data_dir="data/elliptic",
        max_nodes=20,
        ego_radius=1.5,
        train_periods=(1, 34),
        test_periods=(35, 49),
        cache_dir="data/elliptic/processed",
    )

    if args.max_samples is not None:
        use_size = min(args.max_samples, len(test_dataset))
    elif args.quick:
        use_size = min(512, len(test_dataset))
    else:
        use_size = min(5000, len(test_dataset))
    test_dataset = _stratified_subset(test_dataset, use_size, seed=args.seed)
    print(f"[Data] test samples={use_size}")

    raw_test_loader, active_workers = _build_loader(test_dataset, args.batch_size, args.num_workers)
    print(f"[DataLoader] num_workers={active_workers}")

    feature_bounds = None
    if args.clip_to_train_range:
        feature_bounds = _compute_feature_bounds(train_dataset=train_dataset, device=device)
        if feature_bounds is None:
            print("[Attack] train feature range unavailable; fallback to legacy clamp [-10, 10].")
        else:
            print("[Attack] using train-set feature range projection.")

    if args.materialize_quantum:
        cache_path = Path(args.cache_dir) / (
            f"materialized_seed{int(args.seed)}_shots{int(args.n_shots)}_"
            f"ns{int(use_size)}.pt"
        )
        cached_dataset = None
        if args.cache_materialized and cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
                classical = payload["classical"].float().cpu()
                quantum = payload["quantum"].float().cpu()
                labels = payload["labels"].long().cpu()
                cached_dataset = MaterializedAttackDataset(classical=classical, quantum=quantum, labels=labels)
                print(f"[Cache] Reused materialized tensors: {cache_path}")
            except Exception as exc:
                print(f"[Cache] Failed to load materialized tensors ({cache_path}): {exc}. Recomputing...")

        if cached_dataset is None:
            cached_dataset = materialize_quantum_batches(
                model=model,
                raw_loader=raw_test_loader,
                device=device,
                quiet_quantum_logs=quiet_quantum_logs,
            )
            if args.cache_materialized:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "classical": cached_dataset.classical,
                        "quantum": cached_dataset.quantum,
                        "labels": cached_dataset.labels,
                        "meta": {
                            "seed": int(args.seed),
                            "n_shots": int(args.n_shots),
                            "max_samples": int(use_size),
                        },
                    },
                    cache_path,
                )
                print(f"[Cache] Saved materialized tensors: {cache_path}")

        test_loader, mat_workers = _build_materialized_loader(cached_dataset, args.batch_size, args.num_workers)
        print(f"[CacheLoader] num_workers={mat_workers}")
    else:
        test_loader = raw_test_loader

    clean = evaluate_clean(
        model,
        test_loader,
        device,
        threshold=args.decision_threshold,
        quiet_quantum_logs=quiet_quantum_logs,
    )
    print(f"[Clean] acc={clean['accuracy']:.4f}, f1={clean['f1']:.4f}, auc={clean['auc']:.4f}")

    results = {"qgad": {"clean_accuracy": clean["accuracy"], "clean_f1": clean["f1"], "clean_auc": clean["auc"]}}
    rows = []
    attacked_by_eps = evaluate_under_attack_multi_eps(
        model=model,
        test_loader=test_loader,
        epsilons=args.epsilon,
        device=device,
        threshold=args.decision_threshold,
        quiet_quantum_logs=quiet_quantum_logs,
        feature_bounds=feature_bounds,
        attack_feature_ratio=float(args.attack_feature_ratio),
    )
    for eps in args.epsilon:
        attacked = attacked_by_eps[float(eps)]
        drop = (clean["f1"] - attacked["f1"]) / max(clean["f1"], 1e-8) * 100
        print(f"  eps={eps:.3f}: f1={attacked['f1']:.4f} (drop={drop:.2f}%), auc={attacked['auc']:.4f}")
        results["qgad"][f"epsilon_{eps}"] = {
            "accuracy": attacked["accuracy"],
            "f1": attacked["f1"],
            "auc": attacked["auc"],
            "f1_drop_percent": drop,
        }
        rows.append(
            {
                "Method": "Q-GAD",
                "Epsilon": eps,
                "Clean_F1": clean["f1"],
                "Attacked_F1": attacked["f1"],
                "Drop_Percent": drop,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "qgad_robustness_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "adversarial_robustness_standardized",
                "parameters": {
                    "seed": int(args.seed),
                    "n_shots": int(args.n_shots),
                    "decision_threshold": float(args.decision_threshold),
                    "max_samples": int(use_size),
                    "batch_size": int(args.batch_size),
                    "epsilon": [float(e) for e in args.epsilon],
                    "device": str(device),
                    "materialize_quantum": bool(args.materialize_quantum),
                    "clip_to_train_range": bool(args.clip_to_train_range),
                    "cache_materialized": bool(args.cache_materialized),
                    "attack_feature_ratio": float(args.attack_feature_ratio),
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    csv_path = output_dir / "qgad_adversarial_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] {json_path}")
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()
