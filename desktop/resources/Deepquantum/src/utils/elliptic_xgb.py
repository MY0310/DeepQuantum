"""
Shared Elliptic XGBoost training/loading utilities for experiments.
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


DEFAULT_SHARED_XGB_MODEL_PATH = "experiments/shared_models/elliptic_xgboost_model.pkl"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.0


def resolve_xgb_model_path(path_like: Optional[str]) -> Path:
    model_path = Path(path_like or DEFAULT_SHARED_XGB_MODEL_PATH)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    return model_path


def load_xgb_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"XGBoost model not found: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"[XGBoost] Loaded cached model: {model_path}")
    return model


def evaluate_xgb_model_temporal(
    model,
    train_periods: Tuple[int, int] = (1, 34),
    test_periods: Tuple[int, int] = (35, 49),
) -> Dict[str, float]:
    """
    Evaluate cached XGBoost model on Elliptic temporal test split.
    Used as a guard against stale/inverted cached models.
    """
    from data.elliptic_dataset import EllipticPlusPlusDataset

    dataset = EllipticPlusPlusDataset(data_dir=str(PROJECT_ROOT / "data" / "elliptic"))
    if not dataset.load_data():
        raise RuntimeError("Failed to load Elliptic dataset for XGBoost cache sanity check.")

    _, test_nodes = dataset.split_by_time(train_periods=train_periods, test_periods=test_periods)
    if len(test_nodes) == 0:
        raise RuntimeError("No test nodes found for XGBoost cache sanity check.")

    x_df = dataset.features_df.loc[test_nodes]
    y_true = dataset.classes_df.loc[test_nodes, "label"].astype(int).to_numpy(dtype=np.int64)

    y_pred = model.predict(x_df.to_numpy(dtype=np.float32))
    y_prob = model.predict_proba(x_df.to_numpy(dtype=np.float32))[:, 1]

    return {
        "test_samples": int(len(test_nodes)),
        "test_recall_fraud": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "test_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "test_auc": _safe_auc(y_true, y_prob),
        "test_ap": float(average_precision_score(y_true, y_prob)),
    }


def train_elliptic_xgb_model(
    model_path: Path,
    seed: int = 42,
    train_periods: Tuple[int, int] = (1, 34),
    test_periods: Tuple[int, int] = (35, 49),
):
    from xgboost import XGBClassifier
    from data.elliptic_dataset import EllipticPlusPlusDataset

    dataset = EllipticPlusPlusDataset(data_dir=str(PROJECT_ROOT / "data" / "elliptic"))
    if not dataset.load_data():
        raise RuntimeError("Failed to load Elliptic dataset for XGBoost retraining.")

    train_nodes, _ = dataset.split_by_time(train_periods=train_periods, test_periods=test_periods)
    if len(train_nodes) == 0:
        raise RuntimeError("No training nodes found for XGBoost retraining.")

    x_df = dataset.features_df.loc[train_nodes]
    y = dataset.classes_df.loc[train_nodes, "label"].astype(int)

    x_train, x_val, y_train, y_val = train_test_split(
        x_df.to_numpy(dtype=np.float32),
        y.to_numpy(dtype=np.int64),
        test_size=0.2,
        random_state=seed,
        stratify=y.to_numpy(dtype=np.int64),
    )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = float(neg / max(pos, 1))

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(x_train, y_train)

    val_pred = model.predict(x_val)
    val_prob = model.predict_proba(x_val)[:, 1]
    report = {
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "val_recall_fraud": float(recall_score(y_val, val_pred, pos_label=1, zero_division=0)),
        "val_f1": float(f1_score(y_val, val_pred, zero_division=0)),
        "val_auc": _safe_auc(y_val, val_prob),
        "scale_pos_weight": scale_pos_weight,
        "train_periods": [int(train_periods[0]), int(train_periods[1])],
        "test_periods": [int(test_periods[0]), int(test_periods[1])],
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[XGBoost] Retrained and saved: {model_path}")
    print(
        "[XGBoost] Validation: "
        f"recall={report['val_recall_fraud']:.4f}, "
        f"f1={report['val_f1']:.4f}, auc={report['val_auc']:.4f}"
    )
    return model, report


def load_or_train_elliptic_xgb(
    model_path_like: Optional[str] = None,
    seed: int = 42,
    force_retrain: bool = False,
):
    model_path = resolve_xgb_model_path(model_path_like)
    if force_retrain or (not model_path.exists()):
        model, report = train_elliptic_xgb_model(model_path=model_path, seed=seed)
        return model, model_path, report

    model = load_xgb_model(model_path)
    try:
        sanity = evaluate_xgb_model_temporal(model)
        auc = float(sanity.get("test_auc", 0.0))
        ap = float(sanity.get("test_ap", 0.0))
        # Guardrail: an inverted/stale model often has extremely low AUC on temporal test.
        if auc < 0.50 or ap < 0.05:
            print(
                "[XGBoost] Cached model sanity check failed "
                f"(AUC={auc:.4f}, AP={ap:.4f}). Retraining..."
            )
            model, train_report = train_elliptic_xgb_model(model_path=model_path, seed=seed)
            merged_report = {
                "cache_sanity_failed": True,
                "cache_sanity_metrics": sanity,
                "retrained": True,
                **train_report,
            }
            return model, model_path, merged_report
        print(
            "[XGBoost] Cached model sanity check passed "
            f"(AUC={auc:.4f}, AP={ap:.4f})."
        )
        return model, model_path, {"cache_sanity_failed": False, "cache_sanity_metrics": sanity}
    except Exception as exc:
        print(f"[XGBoost] Warning: cache sanity check skipped due to error: {exc}")
        return model, model_path, None
