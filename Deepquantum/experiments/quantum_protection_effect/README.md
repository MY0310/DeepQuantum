# 量子保护效应实验

最后更新：2026-04-23

## 目标

- 在相同分类头下比较 `Q-GAD (Full)` 与 `Q-GAD (ZeroQuantum)` 的抗扰动表现。
- 评估 FGSM（作用于经典特征）下的性能保持率。

## 入口

- 脚本：`experiments/quantum_protection_effect/run_experiment.py`
- 模型：`checkpoints/elliptic_model.pt`
- 后端：真实 DeepQuantum（禁用 mock）

## 运行示例

```bash
cd Deepquantum
python experiments/quantum_protection_effect/run_experiment.py --device cuda --max-samples 256 --batch-size 16 --num-workers 0 --decision-threshold 0.26 --optimize-threshold --epsilon 0.01 0.05 0.10 --n-shots 12
```

说明：
- 建议开启 `--optimize-threshold`，避免固定阈值掩盖真实鲁棒性差异。
- 输出同时包含 `F1` 与 `AUC` 的保持率/下降率。

## 输出

- `experiments/quantum_protection_effect/results/quantum_protection_results.json`
- `experiments/quantum_protection_effect/results/quantum_protection_comparison.csv`

## 最新结果入口

- 汇总记录：`experiments/RESULTS.md`
