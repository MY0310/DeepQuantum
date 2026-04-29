# Q-GAD Experiments

最后更新：2026-04-24

本目录仅保留当前可复现实验入口与结果记录。过时说明已清理。

## 统一要求

- 环境：`D:\Tools\Miniconda3\envs\qgad`
- 量子模型：`checkpoints/elliptic_model.pt`
- 经典共享模型：`experiments/shared_models/elliptic_xgboost_model.pkl`
- 后端要求：真实 DeepQuantum（禁用 mock）
- DataLoader 并行：支持 `--num-workers`，若当前 Windows 会话触发 `WinError 5`，脚本会自动回退 `num_workers=0` 继续运行（不再中断）
- 对抗鲁棒性：支持量子特征物化缓存（`experiments/adversarial_robustness/cache/`），重复运行显著提速
- 消融实验：支持 `--parallel-models` 并行训练（`classical/quantum/hybrid`）

## 实验入口（当前有效）

| 实验 | 目录 | 入口脚本 |
|---|---|---|
| 抗特征伪造 | `feature_forgery_resistance/` | `run.py` |
| 对抗鲁棒性 | `adversarial_robustness/` | `run_adversarial_real.py` |
| 特征敏感性 | `feature_sensitivity/` | `run_experiment.py` |
| 量子保护效应 | `quantum_protection_effect/` | `run_experiment.py` |
| 时序泛化 | `temporal_generalization/` | `run_experiment.py` |
| 拓扑不变量 | `topological_invariance/` | `run_with_qgad_model.py` |
| 消融实验（已改良重跑） | `ablation_study/` | `run.py` |

## 本轮实验范围

- 已重跑：全部实验（含消融实验改良版）
- 统一种子：`seed=42`
- 统一设备：`cuda`
- 通用评估子集：按实验目标采用 `n_samples=64` / `max_samples=128` / `max_samples=256`（消融为 `train/val/test=1200/400/400`）

## 结果入口

- 总记录（唯一指标来源）：`experiments/RESULTS.md`
