# GNN Baseline（已按 2026-04-23 新结果整理）

本目录用于维护 Q-GAD 的经典 GNN 对照实验（GCN / GAT / GraphSAGE / GIN）。

## 当前有效结构（与脚本输出一致）

```text
gnn_baseline/
├── models/
│   ├── gnn_models.py
│   └── gnn_trainer.py
├── utils/
│   └── comparison_analysis.py
├── outputs/
│   └── experiment_YYYYMMDD_HHMMSS/
│       ├── experiment_summary.json
│       ├── table2_metrics.csv
│       ├── gcn/
│       │   ├── gcn_best_model.pt
│       │   ├── gcn_history.json
│       │   ├── gcn_test_metrics.json
│       │   └── gcn_training_curves.png
│       ├── gat/
│       ├── sage/
│       └── gin/
├── run_gnn_baseline.py
├── example_usage.py
├── README.md
└── QUICKSTART.md
```

## 最新基线结果（Elliptic++ 测试集）

来源：`outputs/experiment_*/experiment_summary.json`（取最新时间戳目录）

| 模型 | 参数量 | Accuracy | Precision | Recall | F1 | AUC | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| GCN | 42,155 | 95.9748% | 0.9153 | 0.4192 | 0.5750 | 0.9221 | 0.6733 |
| GAT | 42,542 | 95.7528% | 0.9391 | 0.3703 | 0.5311 | 0.9070 | 0.6598 |
| GraphSAGE | 54,443 | 95.9268% | 0.8620 | 0.4441 | 0.5862 | 0.9071 | 0.6360 |
| GIN | 54,251 | 96.4247% | 0.8454 | 0.5503 | 0.6667 | 0.9131 | 0.6490 |

## 与 Q-GAD 对比要点（按同轮 Table 2）

- Q-GAD 在 `Accuracy/F1/AUC/AP` 上领先（`Accuracy=96.63%`, `F1=0.6675`, `AUC=0.9256`, `AP=0.6924`）。
- GAT 在 `Precision` 上更高（`0.9391` vs Q-GAD `0.9321`），GIN 在 `Recall` 上更高（`0.5503` vs Q-GAD `0.5199`）。
- 结论：Q-GAD 综合指标更优，但 Precision/Recall 仍有优化空间。

## 运行入口

- 单模型：

```bash
python gnn_baseline/run_gnn_baseline.py --model gcn --epochs 10
```

- 全模型：

```bash
python gnn_baseline/run_gnn_baseline.py --model all --epochs 10
```

## 输出规则（已简化）

- 每次运行自动生成：`gnn_baseline/outputs/experiment_YYYYMMDD_HHMMSS/`
- 仅在该目录内保存结果，不再分散生成 `*_checkpoints`、`*_logs` 顶层目录
- 每个模型目录下固定 4 个产物：`*_best_model.pt`、`*_history.json`、`*_test_metrics.json`、`*_training_curves.png`

## 对比分析（可选）

```bash
python gnn_baseline/utils/comparison_analysis.py --quantum-dir experiment_summary.json --classical-dir gnn_baseline/outputs --output-dir gnn_baseline/analysis
```

- `--quantum-dir` 支持目录或 `experiment_summary.json` 文件。

## 说明

- 目录结构可由 `run_gnn_baseline.py` 直接复现，便于后续复用与论文引用。
