# 抗特征伪造实验

最后更新：2026-04-23

单入口脚本：`run.py`

## 目标

1. 对比 `Q-GAD` 与 `XGBoost` 在特征伪造攻击下的鲁棒性。
2. 强制使用真实量子后端（DeepQuantum）；不允许 mock。
3. 使用共享经典模型，避免重复训练并便于跨实验复用。

## 共享模型

- 经典模型训练入口：`run_xgboost.py`
- 默认共享路径：`experiments/shared_models/elliptic_xgboost_model.pkl`

```bash
cd Deepquantum
python run_xgboost.py --model-output experiments/shared_models/elliptic_xgboost_model.pkl --seed 42
```

## 运行实验

```bash
cd Deepquantum
python experiments/feature_forgery_resistance/run.py --n-samples 64 --batch-size 16 --num-workers 2 --optimization-steps 3 --forgery-budget 0.1 --n-shots 15 --decision-threshold 0.26 --device cuda --xgb-model-path experiments/shared_models/elliptic_xgboost_model.pkl
```

常用参数：

- `--n-samples`: 采样的欺诈样本数
- `--optimization-steps`: 攻击优化步数
- `--forgery-budget`: 特征扰动预算（L_inf）
- `--n-shots`: 量子采样次数（速度/方差折中）
- `--seed`: 随机种子
- `--xgb-model-path`: 共享 XGBoost 模型路径
- `--retrain-xgb`: 强制重训 XGBoost（可选）

说明：

- 默认会优先复用共享模型，并做基础有效性校验；仅在需要时重训。
- 当前 Windows 会话下若 `--num-workers > 0` 触发 `WinError 5`，脚本会自动回退 `num_workers=0` 并继续运行。

## 输出

- 结果：`results/feature_forgery_unified_results.json`
- 汇总记录：`experiments/RESULTS.md`
