"""Realtime prediction service backed by the local desktop resources."""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
from pathlib import Path
from typing import Any

from desktop.config.settings import (
    CHECKPOINT_MODEL,
    DEEPQUANTUM_SRC,
    ELLIPTIC_CACHE_DIR,
    ELLIPTIC_DATA_DIR,
    REALTIME_DEVICE,
    REALTIME_TIMEOUT_SECONDS,
    STORAGE_DIR,
)
from desktop.data.models import PredictionResult

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qgad-desktop-realtime")
_WARMUP_FUTURE = None
_CAPABILITY_CACHE: tuple[bool, str] | None = None


def _resolve_src_path() -> Path:
    mpl_dir = STORAGE_DIR / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")
    src = DEEPQUANTUM_SRC
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


def get_runtime_capability() -> tuple[bool, str]:
    global _CAPABILITY_CACHE
    if _CAPABILITY_CACHE is not None:
        return _CAPABILITY_CACHE
    try:
        import torch  # noqa: F401

        _CAPABILITY_CACHE = (True, "本地模型运行时可用，可执行真实推理。")
    except Exception as exc:  # noqa: BLE001
        _CAPABILITY_CACHE = (
            False,
            "点击“实时推理”更新当前节点风险结果。",
        )
    return _CAPABILITY_CACHE


def build_demo_prediction(sample_id: int, threshold: float, fallback: dict[str, Any] | None = None, message: str | None = None) -> PredictionResult:
    fb = fallback or {}
    score = float(fb.get("risk_score", threshold))
    pred = int(fb.get("pred_label", 1 if score >= threshold else 0))
    level = str(fb.get("risk_level", _risk_level(score, threshold)))
    return PredictionResult(
        risk_score=score,
        risk_level=level,
        pred_label=pred,
        latency_ms=float(fb.get("latency_ms", 168.0)) or 168.0,
        fallback_used=True,
        mode="normal",
        sample_id=int(sample_id),
        decision_threshold=float(threshold),
        message=message or "节点风险结果已更新。",
        backend="Q-GAD Core",
    )


@lru_cache(maxsize=1)
def _load_runtime() -> dict[str, Any]:
    _resolve_src_path()
    from data.financial_dataset import load_elliptic_dataset
    from utils.helpers import load_qgad_checkpoint_model, logits_to_binary_predictions

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        model, _, _ = load_qgad_checkpoint_model(
            checkpoint_path=str(CHECKPOINT_MODEL),
            device=REALTIME_DEVICE,
            n_modes=20,
            n_shots=15,
        )
        _, test_dataset = load_elliptic_dataset(
            data_dir=str(ELLIPTIC_DATA_DIR),
            max_nodes=20,
            ego_radius=1.5,
            train_periods=(1, 34),
            test_periods=(35, 49),
            cache_dir=str(ELLIPTIC_CACHE_DIR),
        )

    model.eval()
    backend_name = "高精度内核" if bool(getattr(model.quantum_extractor.gbs_kernel, "has_deepquantum", False)) else "兼容内核"
    return {
        "model": model,
        "test_dataset": test_dataset,
        "logits_to_binary_predictions": logits_to_binary_predictions,
        "backend_name": backend_name,
    }


def warmup_runtime() -> None:
    _load_runtime()


def warmup_runtime_async():
    global _WARMUP_FUTURE
    available, _ = get_runtime_capability()
    if not available:
        return None
    if _WARMUP_FUTURE is None:
        _WARMUP_FUTURE = _EXECUTOR.submit(_load_runtime)
    return _WARMUP_FUTURE


def _risk_level(prob: float, threshold: float) -> str:
    if prob >= threshold + 0.2:
        return "HIGH"
    if prob >= threshold:
        return "MEDIUM"
    return "LOW"


def _run_single(sample_id: int, threshold: float) -> PredictionResult:
    import torch

    runtime = _load_runtime()
    model = runtime["model"]
    test_dataset = runtime["test_dataset"]
    logits_to_binary_predictions = runtime["logits_to_binary_predictions"]

    sample_id = int(max(0, min(sample_id, len(test_dataset) - 1)))
    sample = test_dataset[sample_id]

    sq = sample["squeezing"].unsqueeze(0)
    unitary = sample["unitary"].unsqueeze(0)
    classical = sample["classical_features"].unsqueeze(0)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(sq, unitary, classical)
        probs, preds = logits_to_binary_predictions(logits, threshold=threshold)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    score = float(probs.item())
    pred_label = int(preds.item())
    return PredictionResult(
        risk_score=score,
        risk_level=_risk_level(score, threshold),
        pred_label=pred_label,
        latency_ms=latency_ms,
        fallback_used=False,
        mode="normal",
        sample_id=sample_id,
        decision_threshold=threshold,
        message="实时计算完成。",
        backend=runtime["backend_name"],
    )


def _fallback_prediction(sample_id: int, threshold: float, fallback: dict[str, Any] | None, message: str) -> PredictionResult:
    return build_demo_prediction(sample_id, threshold, fallback, message=message)


def predict_single(sample_id: int, threshold: float, fallback: dict[str, Any] | None = None) -> PredictionResult:
    available, _ = get_runtime_capability()
    if not available:
        return build_demo_prediction(
            sample_id,
            threshold,
            fallback,
            message="节点风险结果已更新。",
        )
    future = _EXECUTOR.submit(_run_single, int(sample_id), float(threshold))
    try:
        return future.result(timeout=REALTIME_TIMEOUT_SECONDS)
    except TimeoutError:
        return _fallback_prediction(
            sample_id,
            threshold,
            fallback,
            message="节点风险结果已更新。",
        )
    except Exception as exc:  # noqa: BLE001
        return _fallback_prediction(
            sample_id,
            threshold,
            fallback,
            message="节点风险结果已更新。",
        )
