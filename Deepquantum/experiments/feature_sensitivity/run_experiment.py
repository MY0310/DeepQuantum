"""
Feature sensitivity analysis (standardized).

Standards aligned with feature_forgery_resistance:
1) Real DeepQuantum backend required (no mock).
2) Fast subset support via --quick / --max-samples.
3) Reproducible outputs under experiment-specific results directory.
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
    get_device,
    load_qgad_checkpoint_model,
    logits_to_binary_predictions,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature Sensitivity Analysis (Standardized)")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--epsilon", type=float, nargs="+", default=[0.01, 0.05, 0.10])
    parser.add_argument("--device", default=None, help="cuda/cpu/mps")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--max-samples", type=int, default=None, help="Max test samples")
    parser.add_argument("--n-shots", type=int, default=15, help="DeepQuantum shots per sample")
    parser.add_argument("--decision-threshold", type=float, default=0.5, help="Fraud decision threshold on P(y=1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-quantum-logs", action="store_true", help="Show verbose DeepQuantum logs")
    parser.add_argument(
        "--output-dir",
        default="experiments/feature_sensitivity/results",
        help="Output directory",
    )
    return parser.parse_args()


def _safe_auc(y_true, y_prob) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.0


def _forward(model, squeezing, unitary, classical, quiet_quantum_logs: bool = True):
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
def evaluate_clean(model, test_loader, device, threshold=0.5, quiet_quantum_logs=True):
    all_preds, all_labels, all_probs = [], [], []
    for batch in test_loader:
        logits = _forward(
            model,
            batch["squeezing"],
            batch["unitary"],
            batch["classical_features"].to(device),
            quiet_quantum_logs=quiet_quantum_logs,
        )
        probs_pos, preds = logits_to_binary_predictions(logits, threshold=threshold)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"].cpu().numpy())
        all_probs.extend(probs_pos.cpu().numpy())
    return {
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "auc": _safe_auc(all_labels, all_probs),
    }


@torch.no_grad()
def evaluate_with_perturbation(
    model,
    test_loader,
    epsilon: float,
    device: str,
    perturb_quantum: bool = False,
    perturb_classical: bool = False,
    quiet_quantum_logs: bool = True,
    threshold: float = 0.5,
):
    all_preds, all_labels, all_probs = [], [], []

    for batch in test_loader:
        squeezing = batch["squeezing"]
        unitary = batch["unitary"]
        classical = batch["classical_features"].to(device)
        labels = batch["label"].to(device)

        if perturb_quantum:
            squeezing = squeezing + torch.randn_like(squeezing) * epsilon
            unitary = unitary + torch.randn_like(unitary) * epsilon
        if perturb_classical:
            classical = classical + torch.randn_like(classical) * epsilon

        logits = _forward(
            model,
            squeezing,
            unitary,
            classical,
            quiet_quantum_logs=quiet_quantum_logs,
        )
        probs_pos, preds = logits_to_binary_predictions(logits, threshold=threshold)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs_pos.cpu().numpy())

    return {
        "f1": float(f1_score(all_labels, all_preds, zero_division=0)),
        "auc": _safe_auc(all_labels, all_probs),
    }


def run_experiment(args: argparse.Namespace):
    set_seed(args.seed)
    device = get_device(args.device)
    quiet_quantum_logs = not args.show_quantum_logs

    print("\n" + "=" * 80)
    print("Feature Sensitivity Analysis (Standardized)")
    print("=" * 80)
    print(
        f"[Config] device={device}, quick={args.quick}, epsilon={args.epsilon}, "
        f"n_shots={args.n_shots}, threshold={args.decision_threshold:.3f}"
    )

    model = load_model(device=device, n_shots=args.n_shots)

    _, test_dataset = load_elliptic_dataset(
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

    test_loader, active_workers = _build_loader(test_dataset, args.batch_size, args.num_workers)
    print(f"[DataLoader] num_workers={active_workers}")

    clean = evaluate_clean(
        model,
        test_loader,
        device,
        threshold=args.decision_threshold,
        quiet_quantum_logs=quiet_quantum_logs,
    )
    print(f"[Clean] F1={clean['f1']:.4f}, AUC={clean['auc']:.4f}")

    scenarios = [
        ("Quantum Features", True, False),
        ("Classical Features", False, True),
        ("Both Features", True, True),
    ]

    results = {"clean": clean}
    rows = []

    for scenario_name, perturb_q, perturb_c in scenarios:
        print(f"\n[Scenario] {scenario_name}")
        for eps in args.epsilon:
            attacked = evaluate_with_perturbation(
                model,
                test_loader,
                epsilon=eps,
                device=device,
                perturb_quantum=perturb_q,
                perturb_classical=perturb_c,
                quiet_quantum_logs=quiet_quantum_logs,
                threshold=args.decision_threshold,
            )
            f1_drop_pct = (clean["f1"] - attacked["f1"]) / max(clean["f1"], 1e-8) * 100
            auc_drop_pct = (clean["auc"] - attacked["auc"]) / max(clean["auc"], 1e-8) * 100 if clean["auc"] > 0 else 0.0

            print(f"  eps={eps:.3f}: F1={attacked['f1']:.4f} (drop={f1_drop_pct:.2f}%), AUC={attacked['auc']:.4f}")

            key = f"{scenario_name.replace(' ', '_')}_epsilon_{eps}"
            results[key] = {
                "f1": attacked["f1"],
                "auc": attacked["auc"],
                "f1_drop_percent": f1_drop_pct,
                "auc_drop_percent": auc_drop_pct,
            }
            rows.append(
                {
                    "Perturbation_Type": scenario_name,
                    "Epsilon": eps,
                    "Clean_F1": clean["f1"],
                    "Attacked_F1": attacked["f1"],
                    "F1_Drop_Percent": f1_drop_pct,
                    "Clean_AUC": clean["auc"],
                    "Attacked_AUC": attacked["auc"],
                    "AUC_Drop_Percent": auc_drop_pct,
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "feature_sensitivity_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "feature_sensitivity_standardized",
                "parameters": {
                    "seed": int(args.seed),
                    "n_shots": int(args.n_shots),
                    "decision_threshold": float(args.decision_threshold),
                    "max_samples": int(use_size),
                    "batch_size": int(args.batch_size),
                    "epsilon": [float(e) for e in args.epsilon],
                    "device": str(device),
                },
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    csv_path = output_dir / "feature_sensitivity_comparison.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] {json_path}")
    print(f"[Saved] {csv_path}")


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
