# Q-GAD Main Project

最后更新：2026-04-24

本目录是 Q-GAD 主工程，包含训练、实验、基线与可视化脚本。

## 核心结构

```text
Deepquantum/
├── src/                         # 核心模块（数据、模型、GBS、工具）
├── experiments/                 # 论文实验与统一结果记录
├── gnn_baseline/                # GCN/GAT/SAGE/GIN 基线
├── visualization/               # 统一可视化入口（稳健 matplotlib 配置）
├── checkpoints/                 # 训练模型（混合模型使用 elliptic_model.pt）
├── outputs/                     # 导出结果/图表
├── run_elliptic.py              # 标准训练入口
├── run_elliptic_fast.py         # 快速训练入口
└── run_xgboost.py               # XGBoost 共享模型训练入口
```

## 统一约定

- 推荐环境：`D:\Tools\Miniconda3\envs\qgad`
- 混合模型文件：`checkpoints/elliptic_model.pt`
- 共享 XGBoost：`experiments/shared_models/elliptic_xgboost_model.pkl`
- 实验总记录：`experiments/RESULTS.md`
- 实验后端：真实 DeepQuantum（禁用 mock）

## 常用命令

```bash
# 1) 刷新经典共享模型
python run_xgboost.py --model-output experiments/shared_models/elliptic_xgboost_model.pkl --seed 42

# 2) 运行特征伪造实验
python experiments/feature_forgery_resistance/run.py --n-samples 64 --batch-size 16 --optimization-steps 3 --forgery-budget 0.1 --n-shots 15 --decision-threshold 0.26 --device cuda --xgb-model-path experiments/shared_models/elliptic_xgboost_model.pkl

# 3) 运行 GNN 基线
python gnn_baseline/run_gnn_baseline.py --model all --epochs 10 --batch-size 32 --device cuda

# 4) 生成论文可视化
python -m visualization.generate_all
```

## 文档入口

- 实验总览：`experiments/README.md`
- 实验结果：`experiments/RESULTS.md`
- GNN 基线：`gnn_baseline/README.md`
- 可视化：`visualization/README.md`
