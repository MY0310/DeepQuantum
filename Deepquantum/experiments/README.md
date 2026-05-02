# Q-GAD Experiments

最后更新：2026-05-03

本目录仅保留当前可复现实验入口与结果记录。

## 统一要求

- 环境：`conda activate qgad`（Python 3.10.20, PyTorch 2.10.0, CUDA 12.8）
- 量子模型：`checkpoints/elliptic_model.pt`
- 经典共享模型：`experiments/shared_models/elliptic_xgboost_model.pkl`
- 量子后端：真实 DeepQuantum（禁用 mock）
- DataLoader：支持 `--num-workers`，Windows 下自动回退 `num_workers=0`
- 对抗鲁棒性：支持量子特征物化缓存（`adversarial_robustness/cache/`），重复运行显著提速
- 消融实验：支持 `--parallel-models` 并行训练

## 实验入口

| # | 实验 | 目录 | 入口脚本 |
|---|---|---|---|
| 1 | 抗特征伪造 | `feature_forgery_resistance/` | `run.py` |
| 2 | 对抗鲁棒性 | `adversarial_robustness/` | `run_adversarial_real.py` |
| 3 | 特征敏感性 | `feature_sensitivity/` | `run_experiment.py` |
| 4 | 量子保护效应 | `quantum_protection_effect/` | `run_experiment.py` |
| 5 | 时序泛化 | `temporal_generalization/` | `run_experiment.py` |
| 6 | 拓扑不变量 | `topological_invariance/` | `run_with_qgad_model.py` |
| 7 | 消融实验 | `ablation_study/` | `run.py` |

## 实验参数基线

- 统一种子：`seed=42`
- 统一设备：`cuda`
- 通用评估子集：按实验目标采用 `n_samples=64` / `max_samples=128` / `max_samples=256`（消融为 `train/val/test=1200/400/400`）

## 缓存文件策略

保留以下缓存（与当前脚本参数完全匹配，可复用量子特征）：

- `adversarial_robustness/cache/materialized_seed42_shots15_ns64.pt`
- `adversarial_robustness/cache/materialized_seed42_shots15_ns128.pt`
- `ablation_study/cache/materialized_seed42_shots8_tr1200_va400_te400.pt`

## 结果入口

- 总记录（唯一指标来源）：`RESULTS.md`
