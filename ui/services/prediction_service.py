"""实时预测与关系图服务。"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ui.config.settings import DEEPQUANTUM_ROOT, REALTIME_DEVICE, REALTIME_TIMEOUT_SECONDS
from ui.data.models import PredictionResult

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qgad-realtime")


def _resolve_src_path() -> Path:
    src = DEEPQUANTUM_ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src


@lru_cache(maxsize=1)
def _load_runtime() -> dict[str, Any]:
    """
    Load model and dataset once.

    Uses CPU mode for reproducible demo runtime.
    """
    _resolve_src_path()
    from data.financial_dataset import load_elliptic_dataset
    from utils.helpers import load_qgad_checkpoint_model, logits_to_binary_predictions

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        model, _, _ = load_qgad_checkpoint_model(
            checkpoint_path=str(DEEPQUANTUM_ROOT / "checkpoints" / "elliptic_model.pt"),
            device=REALTIME_DEVICE,
            n_modes=20,
            n_shots=15,
        )
        _, test_dataset = load_elliptic_dataset(
            data_dir=str(DEEPQUANTUM_ROOT / "data" / "elliptic"),
            max_nodes=20,
            ego_radius=1.5,
            train_periods=(1, 34),
            test_periods=(35, 49),
            cache_dir=str(DEEPQUANTUM_ROOT / "data" / "elliptic" / "processed"),
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
    """Preload runtime resources to reduce first-call latency."""
    _load_runtime()


def _apply_mode_transform(classical: torch.Tensor, mode: str) -> torch.Tensor:
    mode = (mode or "normal").lower()
    c = classical.clone()
    if mode == "forgery":
        center = c.mean(dim=1, keepdim=True)
        c = 0.92 * c + 0.08 * center
    elif mode == "adversarial":
        eps = 0.05
        signed = torch.sign(c)
        c = c + eps * signed * torch.std(c, dim=1, keepdim=True).clamp(min=1e-6)
    return c


def _risk_level(prob: float, threshold: float) -> str:
    if prob >= threshold + 0.2:
        return "HIGH"
    if prob >= threshold:
        return "MEDIUM"
    return "LOW"


def _run_single(sample_id: int, mode: str, threshold: float) -> PredictionResult:
    runtime = _load_runtime()
    model = runtime["model"]
    test_dataset = runtime["test_dataset"]
    logits_to_binary_predictions = runtime["logits_to_binary_predictions"]

    sample_id = int(max(0, min(sample_id, len(test_dataset) - 1)))
    sample = test_dataset[sample_id]

    sq = sample["squeezing"].unsqueeze(0)
    unitary = sample["unitary"].unsqueeze(0)
    classical = sample["classical_features"].unsqueeze(0)
    classical = _apply_mode_transform(classical, mode)

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
        mode=mode,
        sample_id=sample_id,
        decision_threshold=threshold,
        message="实时计算完成。",
        backend=runtime["backend_name"],
    )


def _fallback_prediction(sample_id: int, mode: str, threshold: float, fallback: dict[str, Any] | None, message: str) -> PredictionResult:
    fb = fallback or {}
    score = float(fb.get("risk_score", threshold))
    pred = int(fb.get("pred_label", 1 if score >= threshold else 0))
    level = str(fb.get("risk_level", _risk_level(score, threshold)))
    return PredictionResult(
        risk_score=score,
        risk_level=level,
        pred_label=pred,
        latency_ms=float(fb.get("latency_ms", 0.0)),
        fallback_used=True,
        mode=mode,
        sample_id=int(sample_id),
        decision_threshold=float(threshold),
        message=message,
        backend="本地缓存",
    )


def predict_single(sample_id: int, mode: str, threshold: float, fallback: dict[str, Any] | None = None) -> PredictionResult:
    """
    Public API for realtime single-sample inference.

    Args:
        sample_id: index in test split.
        mode: normal|forgery|adversarial.
        threshold: decision threshold.
        fallback: optional cached values.
    """
    future = _EXECUTOR.submit(_run_single, int(sample_id), str(mode), float(threshold))
    try:
        return future.result(timeout=REALTIME_TIMEOUT_SECONDS)
    except TimeoutError:
        return _fallback_prediction(
            sample_id,
            mode,
            threshold,
            fallback,
            message=f"实时计算超过 {REALTIME_TIMEOUT_SECONDS:.0f} 秒，已切换为缓存结果。",
        )
    except Exception as exc:  # noqa: BLE001
        return _fallback_prediction(
            sample_id,
            mode,
            threshold,
            fallback,
            message=f"实时计算失败（{exc.__class__.__name__}），已切换为缓存结果。",
        )

