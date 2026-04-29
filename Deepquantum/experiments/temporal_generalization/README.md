# 时序泛化实验

最后更新：2026-04-23

## 目标

- 在不同时间间隔配置下评估训练期到测试期的性能保持率。
- 观察 `F1` 与 `AUC` 在时间跨度变化时的退化趋势。

## 入口

- 脚本：`experiments/temporal_generalization/run_experiment.py`
- 模型：`checkpoints/elliptic_model.pt`
- 缓存：统一使用 `data/elliptic/processed`（不再使用 `processed_t*`）

## 时间配置

- `gap_8weeks`: train `(1,40)`, test `(41,49)`
- `gap_15weeks`: train `(1,34)`, test `(35,49)`
- `gap_19weeks`: train `(1,30)`, test `(31,49)`

## 运行示例

```bash
cd Deepquantum
python experiments/temporal_generalization/run_experiment.py --device cuda --configs gap_8weeks gap_15weeks gap_19weeks --max-samples 256 --batch-size 16 --num-workers 0 --decision-threshold 0.26 --optimize-threshold --n-shots 12
```

说明：
- 建议开启 `--optimize-threshold`，可显著降低阈值错配导致的 `F1=0` 假阴性现象。

## 输出

- `experiments/temporal_generalization/results/temporal_generalization_results.json`
- `experiments/temporal_generalization/results/temporal_generalization_comparison.csv`

## 最新结果入口

- 汇总记录：`experiments/RESULTS.md`
