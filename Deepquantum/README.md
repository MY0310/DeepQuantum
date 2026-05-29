# Q-GAD Main Project

最后更新：2026-05-03

本目录是 Q-GAD 主工程，包含训练、实验、基线与可视化脚本。
在仓库文档中本目录统一称为 `app root`（仓库外层目录称为 `repo root`）。

## 核心结构

```text
Deepquantum/
├── src/                         # 核心模块
│   ├── data/                    #   数据集加载（Elliptic、时序图）
│   ├── gbs/                     #   GBS 量子核（基于 DeepQuantum）
│   ├── models/                  #   混合分类器（量子+经典融合）
│   ├── utils/                   #   工具函数、XGBoost、图操作
│   └── trainer.py               #   训练器
├── configs/
│   └── config.py                # ExperimentConfig 配置体系
├── experiments/                 # 7 组论文实验 + shared_models
├── gnn_baseline/                # GCN / GAT / GraphSAGE / GIN 基线
├── visualization/               # 统一可视化入口
├── checkpoints/                 # 训练模型权重（gitignored）
├── outputs/                     # 输出结果 / 图表（gitignored）
├── run_elliptic.py              # 标准训练入口
├── run_elliptic_fast.py         # 快速训练入口
└── run_xgboost.py               # XGBoost 共享模型训练入口
```

## 统一约定

- 推荐环境：`D:\Tools\Miniconda3\envs\qgad`
- 依赖基线：`python=3.10.20`、`torch=2.10.0`、`torchvision=0.26.0`、`numpy=1.26.4`
- 量子框架：`deepquantum==4.5.0`（pip 安装）
- GPU 基线：CUDA 12.8
- 混合模型文件：`checkpoints/elliptic_model.pt`
- 共享 XGBoost：`experiments/shared_models/elliptic_xgboost_model.pkl`
- 实验总记录：`experiments/RESULTS.md`
- 量子后端：真实 DeepQuantum（禁用 mock）

## 常用命令

```bash
conda activate qgad

# 1) 训练 Q-GAD 主模型
python run_elliptic.py

# 2) 训练 XGBoost 共享模型
python run_xgboost.py --model-output experiments/shared_models/elliptic_xgboost_model.pkl --seed 42

# 3) 运行 GNN 基线
python gnn_baseline/run_gnn_baseline.py --model all --epochs 10 --batch-size 32 --device cuda

# 4) 运行论文实验（示例：对抗鲁棒性）
python experiments/adversarial_robustness/run_adversarial_real.py --max-samples 128 --n-shots 15 --device cuda

# 5) 生成全部可视化
python -m visualization.generate_all

# 若当前目录在 repo root（qgad-project/），使用：
# python -m Deepquantum.visualization.generate_all
```

## 文档入口

- 实验总览与入口：`experiments/README.md`
- 实验结果详情：`experiments/RESULTS.md`
- GNN 基线：`gnn_baseline/README.md`
- 可视化说明：`visualization/README.md`

## 清理约定

- 可删除本地缓存：`__pycache__/`、`.mplconfig/`
- 建议保留实验缓存：`experiments/*/cache/*.pt`（用于复用量子特征、加速重跑）
