# GNN Baseline Quickstart（2026-04-23）

## 1) 直接查看当前有效结果

```bash
dir gnn_baseline\outputs
# 选择最新 experiment_YYYYMMDD_HHMMSS 目录后查看：
type gnn_baseline\outputs\<latest_experiment>\table2_metrics.csv
```

## 2) 重新跑单个模型（示例：GCN）

```bash
python gnn_baseline/run_gnn_baseline.py --model gcn --epochs 10 --batch-size 32 --device cpu
```

## 3) 重新跑四个模型

```bash
python gnn_baseline/run_gnn_baseline.py --model all --epochs 10 --batch-size 32 --device cpu
```

运行后会生成：

```text
gnn_baseline/outputs/experiment_YYYYMMDD_HHMMSS/
├── experiment_summary.json
├── gcn/{gcn_best_model.pt,gcn_history.json,gcn_test_metrics.json,gcn_training_curves.png}
├── gat/{...}
├── sage/{...}
└── gin/{...}
```

## 4) 结果归并（仅在外部迁移结果时）

将每个模型的 `*_test_metrics.json` 复制到：

```text
gnn_baseline/outputs/experiment_YYYYMMDD_HHMMSS/{gcn,gat,sage,gin}/
```

并更新：

- `experiment_summary.json`
- `table2_metrics.csv`

当前标准目录命名：`gnn_baseline/outputs/experiment_YYYYMMDD_HHMMSS/`。

## 5) 量子与GNN对比分析（可选）

```bash
python gnn_baseline/utils/comparison_analysis.py --quantum-dir experiment_summary.json --classical-dir gnn_baseline/outputs --output-dir gnn_baseline/analysis
```
