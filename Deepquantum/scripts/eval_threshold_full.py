"""
High-resource evaluation run:
- Export Q-GAD probabilities on validation/test splits
- Calibrate best F1 threshold on validation split
- Recompute test metrics at threshold=0.5 and threshold=best

This script does NOT retrain the model.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_MPL_DIR = ROOT / ".mplconfig"
DEFAULT_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPL_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

from data.financial_dataset import collate_fn, load_elliptic_dataset
from utils.helpers import find_best_f1_threshold, load_qgad_checkpoint_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("High-resource threshold evaluation (no retraining)")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--worker-device",
        default="",
        help="Device used inside parallel workers. Empty means: cpu when parallel_workers>1, else --device.",
    )
    p.add_argument("--n-shots", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--parallel-workers", type=int, default=1)
    p.add_argument(
        "--worker-threads",
        type=int,
        default=0,
        help="Threads per parallel worker. 0 = auto split from --threads.",
    )
    p.add_argument(
        "--parallel-chunk-batches",
        type=int,
        default=8,
        help="Number of mini-batches per parallel task chunk.",
    )
    p.add_argument("--num-shards", type=int, default=1, help="Total shard count for manual parallel runs.")
    p.add_argument("--shard-id", type=int, default=0, help="Current shard id in [0, num_shards).")
    p.add_argument(
        "--merge-num-shards",
        type=int,
        default=0,
        help="If >0, merge shard outputs and compute final metrics without inference.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-samples", type=int, default=2048)
    p.add_argument("--test-samples", type=int, default=0, help="0 means full test set")
    p.add_argument(
        "--output-dir",
        default="outputs/threshold_eval",
        help="Output directory for probs/summary",
    )
    return p.parse_args()


def set_runtime_threads(threads: int) -> None:
    t = str(int(max(1, threads)))
    os.environ["OMP_NUM_THREADS"] = t
    os.environ["MKL_NUM_THREADS"] = t
    os.environ["OPENBLAS_NUM_THREADS"] = t
    os.environ["NUMEXPR_NUM_THREADS"] = t
    torch.set_num_threads(int(t))


@torch.no_grad()
def collect_probs(
    model,
    loader: DataLoader,
    device: str,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = []
    y_prob = []
    sink = io.StringIO()
    for batch in tqdm(loader, desc=desc):
        # Keep quantum params on CPU for real DeepQuantum path.
        squeezing = batch["squeezing"]
        unitary = batch["unitary"]
        classical = batch["classical_features"].to(device, non_blocking=True)

        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            logits = model(squeezing, unitary, classical)
        probs = torch.softmax(logits, dim=1)[:, 1]

        y_true.extend(batch["label"].cpu().numpy().tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_prob, dtype=np.float32)


def split_indices(indices: list[int], n_parts: int) -> list[list[int]]:
    if n_parts <= 1:
        return [indices]
    arr = np.asarray(indices, dtype=np.int64)
    return [chunk.tolist() for chunk in np.array_split(arr, n_parts) if len(chunk) > 0]


def merge_shard_artifacts(output_dir: Path, num_shards: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_val_parts = []
    p_val_parts = []
    y_test_parts = []
    p_test_parts = []
    for i in range(int(num_shards)):
        shard_path = output_dir / f"qgad_probs_shard{i}_of{num_shards}.npz"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing shard artifact: {shard_path}")
        data = np.load(shard_path)
        y_val_parts.append(data["y_val"])
        p_val_parts.append(data["p_val"])
        y_test_parts.append(data["y_test"])
        p_test_parts.append(data["p_test"])

    y_val = np.concatenate(y_val_parts, axis=0)
    p_val = np.concatenate(p_val_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)
    p_test = np.concatenate(p_test_parts, axis=0)
    return y_val, p_val, y_test, p_test


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= float(threshold)).astype(np.int64)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    main_mpl_dir = ROOT / ".mplconfig" / "threshold_main"
    main_mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(main_mpl_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    set_runtime_threads(args.threads)

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    parallel_workers = int(max(1, args.parallel_workers))
    worker_device = args.worker_device.strip() or ("cpu" if parallel_workers > 1 else args.device)
    num_shards = int(max(1, args.num_shards))
    shard_id = int(args.shard_id)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"--shard-id must be in [0, {num_shards})")

    if args.worker_threads > 0:
        worker_threads = int(args.worker_threads)
    else:
        worker_threads = int(max(1, args.threads // parallel_workers))

    print(
        f"[Config] device={args.device}, worker_device={worker_device}, n_shots={args.n_shots}, "
        f"batch={args.batch_size}, workers={args.num_workers}, threads={args.threads}, "
        f"parallel_workers={parallel_workers}, worker_threads={worker_threads}, "
        f"num_shards={num_shards}, shard_id={shard_id}, merge_num_shards={args.merge_num_shards}"
    )

    t0 = time.time()
    if int(args.merge_num_shards) > 0:
        y_val, p_val, y_test, p_test = merge_shard_artifacts(out_dir, int(args.merge_num_shards))
        val_count = int(len(y_val))
        test_count = int(len(y_test))
        print(f"[Merge] merged_shards={args.merge_num_shards}, val={val_count}, test={test_count}")
    else:
        model, _, inferred = load_qgad_checkpoint_model(
            checkpoint_path=str(ROOT / "checkpoints" / "elliptic_model.pt"),
            device=args.device,
            n_modes=20,
            n_shots=args.n_shots,
        )
        has_real = bool(getattr(model.quantum_extractor.gbs_kernel, "has_deepquantum", False))
        if not has_real:
            raise RuntimeError("DeepQuantum backend unavailable. Mock backend is not allowed.")
        print(f"[Model] hidden_dims={inferred['hidden_dims']} real_deepquantum={has_real}")

        train_ds, test_ds = load_elliptic_dataset(
            data_dir=str(ROOT / "data" / "elliptic"),
            max_nodes=20,
            ego_radius=1.5,
            train_periods=(1, 34),
            test_periods=(35, 49),
            cache_dir=str(ROOT / "data" / "elliptic" / "processed"),
        )

        n_train = int(0.8 * len(train_ds))
        perm = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(args.seed)).tolist()
        val_indices = perm[n_train:]
        if args.val_samples > 0:
            val_indices = val_indices[: min(int(args.val_samples), len(val_indices))]
        test_indices = list(range(len(test_ds)))
        if args.test_samples > 0:
            test_indices = test_indices[: min(int(args.test_samples), len(test_indices))]

        if num_shards > 1:
            val_shards = split_indices(val_indices, num_shards)
            test_shards = split_indices(test_indices, num_shards)
            val_indices = val_shards[shard_id]
            test_indices = test_shards[shard_id]

        print(f"[Data] val={len(val_indices)}, test={len(test_indices)}")
        val_loader = DataLoader(
            Subset(train_ds, val_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        test_loader = DataLoader(
            Subset(test_ds, test_indices),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
            persistent_workers=(args.num_workers > 0),
        )
        y_val, p_val = collect_probs(model, val_loader, args.device, desc="Collect/Val")
        y_test, p_test = collect_probs(model, test_loader, args.device, desc="Collect/Test")

        if num_shards > 1:
            shard_npz = out_dir / f"qgad_probs_shard{shard_id}_of{num_shards}.npz"
            np.savez_compressed(
                shard_npz,
                y_val=y_val,
                p_val=p_val,
                y_test=y_test,
                p_test=p_test,
            )
            shard_summary = {
                "timestamp": ts,
                "mode": "shard_only",
                "shard_id": shard_id,
                "num_shards": num_shards,
                "val_samples": int(len(y_val)),
                "test_samples": int(len(y_test)),
                "artifact": str(shard_npz),
                "elapsed_minutes": round((time.time() - t0) / 60.0, 2),
            }
            shard_summary_path = out_dir / f"threshold_shard_summary_{shard_id}_of{num_shards}_{ts}.json"
            with open(shard_summary_path, "w", encoding="utf-8") as f:
                json.dump(shard_summary, f, indent=2, ensure_ascii=False)
            print(f"[ShardDone] summary={shard_summary_path}")
            print(json.dumps(shard_summary, ensure_ascii=False, indent=2))
            return

        val_count = int(len(val_indices))
        test_count = int(len(test_indices))

    best = find_best_f1_threshold(
        y_val,
        p_val,
        min_threshold=0.05,
        max_threshold=0.95,
        num_thresholds=181,
    )
    m05 = metrics_at_threshold(y_test, p_test, 0.5)
    mb = metrics_at_threshold(y_test, p_test, best["threshold"])

    npz_path = out_dir / f"qgad_probs_{ts}.npz"
    np.savez_compressed(
        npz_path,
        y_val=y_val,
        p_val=p_val,
        y_test=y_test,
        p_test=p_test,
    )
    summary = {
        "timestamp": ts,
        "config": {
            "device": args.device,
            "worker_device": worker_device,
            "n_shots": int(args.n_shots),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "threads": int(args.threads),
            "parallel_workers": int(parallel_workers),
            "worker_threads": int(worker_threads),
            "seed": int(args.seed),
            "val_samples": int(val_count),
            "test_samples": int(test_count),
            "num_shards": int(num_shards),
            "shard_id": int(shard_id),
            "merge_num_shards": int(args.merge_num_shards),
        },
        "best_on_val": best,
        "test_at_0.5": m05,
        "test_at_best": mb,
        "delta_f1": float(mb["f1"] - m05["f1"]),
        "artifacts": {
            "npz": str(npz_path),
        },
        "elapsed_minutes": round((time.time() - t0) / 60.0, 2),
    }
    summary_path = out_dir / f"threshold_summary_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Done] summary={summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
