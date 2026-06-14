# Q-GAD Project

最后更新：2026-05-30

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

- Conda 环境：`qgad`（Python 3.10.20, PyTorch 2.10.0 CUDA 12.8）
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

# 或者留在 repo root 直接执行：
# python -m Deepquantum.visualization.generate_all
# python -m Deepquantum.visualization.model_comparison
# python -m Deepquantum.visualization.experiments_dashboard
```

## 文档索引

- 工程总览：`Deepquantum/README.md`
- 实验总览：`Deepquantum/experiments/README.md`
- 实验结果：`Deepquantum/experiments/RESULTS.md`
- GNN 基线：`Deepquantum/gnn_baseline/README.md`
- 可视化说明：`Deepquantum/visualization/README.md`

## 模型主线说明（避免概念混淆）

- Q-GAD 主训练链路是 `PyTorch` 混合模型：`QuantumFeatureExtractor + HybridNeuralClassifier`。
- 训练策略是交替优化：先训练量子核，再训练混合神经网络分类头。
- `XGBoost` 在本仓库中主要用于纯经典基线（`run_xgboost.py`）。
- `XGBoost` 也用于可选后融合（先提取特征，再单独训练 `XGBoost`），不是主训练默认头。
- 可预计算并缓存的是图编码参数 `squeezing/unitary`（来自子图分解），不是最终 9 维量子特征本身。

## 代码用途速查（按模块）

### 1) 训练入口与主流程

- `Deepquantum/run_elliptic.py`：标准训练入口（数据加载、训练、评估、保存）。
- `Deepquantum/run_elliptic_fast.py`：快速验证入口（小规模样本、较快迭代）。
- `Deepquantum/src/trainer.py`：交替训练主逻辑、评估指标、checkpoint/history。
- `Deepquantum/src/models/hybrid_classifier.py`：Q-GAD 主模型定义（量子分支+经典融合头）与可选 `XGBoostFusionClassifier`。
- `Deepquantum/src/gbs/gbs_kernel.py`：GBS 电路构建、采样、量子统计特征提取。

### 2) 数据加载与预处理

- `Deepquantum/src/data/elliptic_dataset.py`：Elliptic++ 原始 CSV 读入、标签清洗、按时段切分。
- `Deepquantum/src/data/financial_dataset.py`：PyTorch `Dataset` 封装、量子参数缓存、`collate_fn`。
- `Deepquantum/src/utils/graph_utils.py`：图预处理核心（ego-net、归一化、Takagi/SVD、参数映射）。
- `Deepquantum/src/data/temporal_graph.py`：时序图扩展数据集（研究扩展用，主线默认不依赖）。

### 3) 经典基线与对比分析

- `Deepquantum/run_xgboost.py`：纯经典特征 `XGBoost` 基线训练与评估。
- `Deepquantum/src/utils/elliptic_xgb.py`：共享 `XGBoost` 模型加载/重训/时序一致性校验。
- `Deepquantum/gnn_baseline/run_gnn_baseline.py`：GCN/GAT/GraphSAGE/GIN 基线训练。
- `Deepquantum/gnn_baseline/models/gnn_models.py`：GNN 架构定义。
- `Deepquantum/gnn_baseline/models/gnn_trainer.py`：GNN 训练循环与指标。
- `Deepquantum/gnn_baseline/utils/comparison_analysis.py`：Q-GAD vs GNN 指标与报告分析。

### 4) 实验与评估脚本

- `Deepquantum/experiments/*/run*.py`：7组实验入口（见下一节）。
- `Deepquantum/scripts/eval_threshold_full.py`：阈值评估与概率导出脚本（不重训模型）。

### 5) 可视化与 UI

- `Deepquantum/visualization/generate_all.py`：统一出图入口。
- `Deepquantum/visualization/model_comparison.py`：Q-GAD 与基线模型综合对比图。
- `Deepquantum/visualization/experiments_dashboard.py`：实验叙事图（鲁棒性、机制、证据）。
- `Deepquantum/visualization/data_sources.py`：统一读取结果文件，避免绘图脚本硬编码。
- `ui/app.py`：Streamlit 单页监控台入口（时段回放、风险子图、节点详情）。

## 七组实验设计思路（做什么、怎么做）

| 实验 | 入口 | 核心问题 | 设计要点 |
|---|---|---|---|
| 抗特征伪造 | `experiments/feature_forgery_resistance/run.py` | 在预算受限伪造下，Recall 会掉多少 | 对经典特征做有限预算扰动，比较 Q-GAD 与 XGBoost 伪造前后 Recall drop |
| 对抗鲁棒性 | `experiments/adversarial_robustness/run_adversarial_real.py` | 不同攻击强度下性能退化轨迹 | 扫 `epsilon=[0.01,0.05,0.1]`，统计 F1/AUC 退化；支持量子特征物化缓存 |
| 特征敏感性 | `experiments/feature_sensitivity/run_experiment.py` | 模型更依赖量子特征还是经典特征 | 分别扰动 quantum/classical/both 三通道，比较 F1/AUC 变化 |
| 量子保护贡献 | `experiments/quantum_protection_effect/run_experiment.py` | 量子分支在攻击下是否带来净增益 | 对比 `Q-GAD (Full)` 与 `Q-GAD (ZeroQuantum)` 的保持率差异 |
| 时序泛化 | `experiments/temporal_generalization/run_experiment.py` | 跨时间窗口迁移是否稳定 | 多组 train/test 时段间隔配置，计算 `retention_rate` 与退化量 |
| 拓扑不变量 | `experiments/topological_invariance/run_with_qgad_model.py` | 是否能区分同构/非同构结构 | 生成图对，提量子特征，算相似度分离度（isomorphic vs non-isomorphic） |
| 消融实验 | `experiments/ablation_study/run.py` + `train_ablation_complete.py` | 各分支真实贡献是什么 | 训练 `Classical-only / Quantum-only / Hybrid`，统一口径比较 F1/AUC |

## 关键参数解释（实验与训练常用）

### 1) 数据与切分

- `train_periods/test_periods`：Elliptic++ 时间切分区间（时序泛化与主训练关键）。
- `max_nodes`：每个 ego 子图最大节点数（对应量子模式上限）。
- `ego_radius`：子图提取半径（通常 `1.5`，实现上近似 2-hop 语义）。

### 2) 量子相关

- `n_modes`：量子模式数，通常与 `max_nodes` 一致或更小（快速模式）。
- `n_shots`：每个样本采样次数，越大方差越低但耗时越高。
- `squeezing/unitary`：图结构编码参数，可预计算缓存复用。

### 3) 训练与评估

- `quantum_epochs/hybrid_epochs`：交替训练中两个阶段的每轮步数。
- `decision_threshold`：将 `P(y=1)` 转为标签的阈值，直接影响 Precision/Recall 取舍。
- `optimize_threshold`：是否在验证集搜索最优 F1 阈值后再评测测试集。

### 4) 攻击与鲁棒性

- `epsilon`：扰动强度（常见 `0.01/0.05/0.1`）。
- `forgery_budget`：特征伪造预算（常用 `L∞` 约束）。
- `optimization_steps`：伪造攻击优化步数，越多通常攻击越强但更慢。
- `attack_feature_ratio`：参与攻击的特征占比（1.0 表示全部特征）。

### 5) 采样与效率

- `max_samples`：实验评估样本上限，用于控时和可复现。
- `num_workers`：DataLoader 并行线程，Windows 下常回退到 0。
- `materialize_quantum/cache_materialized`：是否一次性物化并缓存量子特征以加速重复评估。

## 关于“文件较多”的说明

- 主训练最短链路其实很清晰：`run_elliptic.py -> financial_dataset.py/graph_utils.py -> trainer.py -> hybrid_classifier.py/gbs_kernel.py`。
- 额外文件主要服务于三件事：实验复现、跨方法对照（GNN/XGBoost）、论文与答辩可视化输出。
- 因此“文件多”更多是科研工程化拆分，不代表主链路复杂度失控。

## Git 注意事项

- `data/`, `*.pt`, `*.pkl`, `outputs/` 等大文件已加入 `.gitignore`
- `experiments/*/cache/` 保留在本地用于实验复用，不提交到远程
- 历史中的 `features.csv`（658MB）已通过 `git filter-repo` 移除
