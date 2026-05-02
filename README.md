# Q-GAD Project

最后更新：2026-05-03

基于 DeepQuantum 框架与高斯玻色采样 (GBS) 的金融反欺诈图异常检测系统。

## 命名说明

- 外层 `qgad-project/`：仓库根（repo root）
- 内层 `Deepquantum/`：主工程根（app root）

## 目录结构

```text
.
├── Deepquantum/              # 主工程
│   ├── src/                  # 核心源码
│   ├── configs/              # 配置
│   ├── experiments/          # 8 组实验（消融、鲁棒性、拓扑等）
│   ├── gnn_baseline/         # GNN 基线对比
│   ├── visualization/        # 可视化生成
│   ├── data/                 # 数据集（gitignored）
│   ├── checkpoints/          # 模型权重（gitignored）
│   └── outputs/              # 输出图表（gitignored）
├── article/                  # 论文与材料（gitignored）
├── ui/                       # UI 原型（gitignored）
└── README.md
```

## 环境要求

- Conda 环境：`qgad`（Python 3.12.8, PyTorch 2.10.0 CUDA 12.8）
- 关键依赖：PyTorch, PennyLane, scikit-learn, XGBoost, NetworkX, Matplotlib

## 快速开始

```powershell
conda activate qgad
cd Deepquantum

# 训练主模型
python run_elliptic.py

# 运行 GNN 基线对比
python gnn_baseline/run_gnn_baseline.py

# 生成全部可视化
python -m visualization.generate_all
```

## 文档索引

- 工程总览：`Deepquantum/README.md`
- 实验总览：`Deepquantum/experiments/README.md`
- 实验结果：`Deepquantum/experiments/RESULTS.md`
- GNN 基线：`Deepquantum/gnn_baseline/README.md`
- 可视化说明：`Deepquantum/visualization/README.md`

## Git 注意事项

- `data/`, `*.pt`, `*.pkl`, `outputs/` 等大文件已加入 `.gitignore`
- `experiments/*/cache/` 保留在本地用于实验复用，不提交到远程
- 历史中的 `features.csv`（658MB）已通过 `git filter-repo` 移除
