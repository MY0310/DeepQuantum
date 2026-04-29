"""
Temporal generalization experiment (standardized).

Standards:
1) Real DeepQuantum backend required.
2) Fast subset support and reproducible config.
3) Clear per-config retention/degradation outputs.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from data.financial_dataset import collate_fn, load_elliptic_dataset
from utils.helpers import (
    find_best_f1_threshold,
    get_device,
    load_qgad_checkpoint_model,
    logits_to_binary_predictions,
    set_seed,
)


TIME_CONFIGS = {
    "gap_8weeks": {"train": (1, 40), "test": (41, 49), "gap_weeks": 8, "description": "中等间隔（8周）"},
    "gap_15weeks": {"train": (1, 34), "test": (35, 49), "gap_weeks": 15, "description": "标准分割（15周）"},
    "gap_19weeks": {"train": (1, 30), "test": (31, 49), "gap_weeks": 19, "description": "最大间隔（19周）"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal Generalization (Standardized)")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["gap_8weeks", "gap_15weeks", "gap_19weeks"],
        choices=list(TIME_CONFIGS.keys()),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--n-shots", type=int, default=15)
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Fraud decision threshold on P(y=1)")
    parser.add_argument("--optimize-threshold", dest="optimize_threshold", action="store_true")
    parser.add_argument("--no-optimize-threshold", dest="optimize_threshold", action="store_false")
    parser.set_defaults(optimize_threshold=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-quantum-logs", action="store_true")
    parser.add_argument("--output-dir", default="experiments/temporal_generalization/results")
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


def load_model(device: str, n_shots: int):
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
def _collect_labels_probs(model, data_loader, device, quiet_quantum_logs=True):
    all_labels, all_probs = [], []
    for batch in data_loader:
        logits = _forward(
            model,
            batch["squeezing"],
            batch["unitary"],
            batch["classical_features"].to(device),
            quiet_quantum_logs=quiet_quantum_logs,
        )
        probs_pos, _ = logits_to_binary_predictions(logits, threshold=0.5)
        all_labels.extend(batch["label"].cpu().numpy())
        all_probs.extend(probs_pos.cpu().numpy())
    return np.asarray(all_labels), np.asarray(all_probs)


def _evaluate_from_labels_probs(all_labels, all_probs, threshold=0.5):
    preds = (all_probs >= float(threshold)).astype(int)
    return {
        "f1": float(f1_score(all_labels, preds, zero_division=0)),
        "auc": _safe_auc(all_labels, all_probs),
    }


def run_single_config(model, config_name, config, device, args, quiet_quantum_logs=True):
    print(f"\n[Config] {config_name} ({config['description']})")
    print(f"  train_periods={config['train']}, test_periods={config['test']}")

    train_dataset, test_dataset = load_elliptic_dataset(
        data_dir="data/elliptic",
        max_nodes=20,
        ego_radius=1.5,
        train_periods=config["train"],
        test_periods=config["test"],
        cache_dir="data/elliptic/processed",
    )

    if args.max_samples is not None:
        max_size = args.max_samples
    elif args.quick:
        max_size = 512
    else:
        max_size = None

    if max_size is not None:
        train_size = min(max_size, len(train_dataset))
        test_size = min(max_size, len(test_dataset))
        train_dataset = _stratified_subset(train_dataset, train_size, seed=args.seed)
        test_dataset = _stratified_subset(test_dataset, test_size, seed=args.seed + 1)

    train_loader, train_workers = _build_loader(train_dataset, args.batch_size, args.num_workers)
    test_loader, test_workers = _build_loader(test_dataset, args.batch_size, args.num_workers)
    print(f"[DataLoader] train_workers={train_workers}, test_workers={test_workers}")

    train_labels, train_probs = _collect_labels_probs(
        model,
        train_loader,
        device,
        quiet_quantum_logs=quiet_quantum_logs,
    )
    test_labels, test_probs = _collect_labels_probs(
        model,
        test_loader,
        device,
        quiet_quantum_logs=quiet_quantum_logs,
    )

    fixed_threshold = float(args.decision_threshold)
    if args.optimize_threshold:
        best = find_best_f1_threshold(train_labels, train_probs)
        threshold = float(best["threshold"])
        print(
            f"  threshold(train-opt): {threshold:.3f} "
            f"(train_f1={best['f1']:.4f}, precision={best['precision']:.4f}, recall={best['recall']:.4f})"
        )
    else:
        threshold = fixed_threshold
        print(f"  threshold(fixed): {threshold:.3f}")

    train_metrics = _evaluate_from_labels_probs(train_labels, train_probs, threshold=threshold)
    test_metrics = _evaluate_from_labels_probs(test_labels, test_probs, threshold=threshold)
    test_metrics_fixed = _evaluate_from_labels_probs(test_labels, test_probs, threshold=fixed_threshold)

    retention_rate = (test_metrics["f1"] / max(train_metrics["f1"], 1e-8)) * 100
    f1_degradation = train_metrics["f1"] - test_metrics["f1"]
    auc_degradation = train_metrics["auc"] - test_metrics["auc"]

    print(
        f"  train_f1={train_metrics['f1']:.4f}, test_f1={test_metrics['f1']:.4f}, "
        f"retention={retention_rate:.2f}%, f1_deg={f1_degradation:.4f}"
    )

    return {
        "config_name": config_name,
        "description": config["description"],
        "train_periods": list(config["train"]),
        "test_periods": list(config["test"]),
        "gap_weeks": config["gap_weeks"],
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
        "threshold_mode": "train_optimized" if args.optimize_threshold else "fixed",
        "threshold": threshold,
        "fixed_threshold": fixed_threshold,
        "train_f1": train_metrics["f1"],
        "train_auc": train_metrics["auc"],
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc"],
        "test_f1_fixed_threshold": test_metrics_fixed["f1"],
        "retention_rate": retention_rate,
        "f1_degradation": f1_degradation,
        "auc_degradation": auc_degradation,
    }


def run_experiment(args: argparse.Namespace):
    set_seed(args.seed)
    device = get_device(args.device)
    quiet_quantum_logs = not args.show_quantum_logs

    print("\n" + "=" * 80)
    print("Temporal Generalization (Standardized)")
    print("=" * 80)
    print(
        f"[Config] device={device}, configs={args.configs}, quick={args.quick}, "
        f"n_shots={args.n_shots}, threshold={args.decision_threshold:.3f}, "
        f"optimize_threshold={args.optimize_threshold}"
    )

    model = load_model(device=device, n_shots=args.n_shots)

    results = []
    for name in args.configs:
        results.append(
            run_single_config(
                model,
                name,
                TIME_CONFIGS[name],
                device,
                args,
                quiet_quantum_logs=quiet_quantum_logs,
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "temporal_generalization_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "temporal_generalization_standardized",
                "parameters": {
                    "seed": int(args.seed),
                    "n_shots": int(args.n_shots),
                    "decision_threshold": float(args.decision_threshold),
                    "optimize_threshold": bool(args.optimize_threshold),
                    "batch_size": int(args.batch_size),
                    "max_samples": args.max_samples,
                    "quick": bool(args.quick),
                    "configs": args.configs,
                    "device": str(device),
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    csv_path = output_dir / "temporal_generalization_comparison.csv"
    pd.DataFrame(results).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] {json_path}")
    print(f"[Saved] {csv_path}")

    if len(results) >= 2:
        rs = sorted(results, key=lambda x: x["gap_weeks"])
        gaps = np.array([x["gap_weeks"] for x in rs], dtype=float)
        degs = np.array([x["f1_degradation"] for x in rs], dtype=float)
        slope = float(np.polyfit(gaps, degs, deg=1)[0])
        corr = float(np.corrcoef(gaps, degs)[0, 1]) if len(gaps) > 1 else 0.0
        print(f"[Trend] degradation slope={slope:.6f} F1/week, corr={corr:.4f}")


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
