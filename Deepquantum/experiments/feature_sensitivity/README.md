# 特征敏感性实验

最后更新：2026-04-23

## 目标

- 对比量子特征、经典特征、同时扰动三种场景。
- 评估 Q-GAD 的 `F1/AUC` 变化趋势。

## 入口

- 脚本：`experiments/feature_sensitivity/run_experiment.py`
- 模型：`checkpoints/elliptic_model.pt`
- 后端：真实 DeepQuantum（禁用 mock）

## 运行示例

```bash
cd Deepquantum
python experiments/feature_sensitivity/run_experiment.py --device cuda --max-samples 128 --batch-size 16 --num-workers 2 --decision-threshold 0.26 --epsilon 0.01 0.05 0.10 --n-shots 15
```

说明：
- 当前 Windows 会话下若 `--num-workers > 0` 触发 `WinError 5`，脚本会自动回退 `num_workers=0` 并继续运行。

## 输出

- `experiments/feature_sensitivity/results/feature_sensitivity_results.json`
- `experiments/feature_sensitivity/results/feature_sensitivity_comparison.csv`

## 最新结果入口

- 汇总记录：`experiments/RESULTS.md`
