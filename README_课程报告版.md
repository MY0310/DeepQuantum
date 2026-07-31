# 基于变分高斯玻色采样的鲁棒图异常检测系统

> 一个面向隐蔽网络威胁感知的量子-AI 融合图异常检测系统

- **仓库地址**：https://github.com/MY0310/Q-GAD.git
- **作者**：杨焌铃
- **邮箱**：mycqyjl@nuaa.edu.cn

## 项目简介

Q-GAD（Quantum Graph Anomaly Detector）是一个面向隐蔽网络威胁感知的量子-AI 融合图异常检测系统。项目依托图灵量子 DeepQuantum 框架，将高斯玻色采样（Gaussian Boson Sampling, GBS）的全局物理计算特质与深度学习架构结合，构建用于异常检测、鲁棒识别与机制分析的混合防线。

系统通过 Ego-Net 采样与 Takagi-Autonne 分解，将宏观离散的威胁交互图谱映射为连续变量量子态参数；再基于自动微分的变分量子特征核，利用 GBS 采样概率对应的哈夫尼安（Hafnian）物理先验，主动学习高阶异常拓扑指纹。

从实现上看，Q-GAD 由量子特征核、混合神经网络分类器和一套完整的数据处理与实验流程组成，能够在统一框架下完成特征提取、分类训练和结果评估；同时也实现了和 GNN/XGBoost 的对照测试，在相同数据设置下进行比较：

- 数据层通过 Ego-Net 采样提取局部威胁子图，并完成时序切分与图结构整理
- 量子层将局部子图映射为连续变量量子线路参数，生成 `squeezing` 与 `unitary`
- 特征层基于 GBS 采样概率构造量子统计特征，并学习高阶异常拓扑指纹
- 融合层将量子拓扑特征与经典局部属性拼接，输入下游分类器
- 实验层在对抗、伪造、跨时序与拓扑不变量场景下检验模型的鲁棒性

系统的核心思想是利用 GBS 对稠密拓扑结构的敏感性，将威胁图谱中的局部高风险模式映射为可学习的量子特征，再由前馈神经网络完成最终分类。相比仅依赖纯经典特征的模型，Q-GAD 引入了图结构信息与量子采样统计信号，能够更好地刻画伪装性强、结构复杂的异常模式。

项目包含以下能力：

- 主训练流程：训练 Q-GAD 混合模型
- 经典对照：XGBoost 与 GNN
- 论文实验：消融、鲁棒性、泛化性与拓扑不变量分析
- 可视化生成：输出图表与结果汇总
- 监控原型：提供离线风险监控页面
- 阈值评估：对验证集搜索最优判定阈值并生成 test 概率文件

## 环境与依赖

### 运行环境

| 项目 | 版本 | 说明 |
|------|------|------|
| 操作系统 | Windows 10 / Windows 11 / Ubuntu 20.04+ | 开发与运行环境 |
| Python | 3.10.20 | 项目主语言 |
| PyTorch | 2.10.0 | 模型训练与自动微分 |
| torchvision | 0.26.0 | 与 PyTorch 兼容的视觉工具包 |
| CUDA | 12.8 | GPU 加速支持 |
| NumPy | 1.26.4 | 数值计算 |
| Pandas | 2.3.3 | 表格数据处理 |
| SciPy | 1.15.2 | 科学计算与线性代数 |
| scikit-learn | 1.7.2 | 传统机器学习与评估指标 |
| NetworkX | 3.4.2 | 图结构构建与分析 |
| Matplotlib | 3.8.4 | 绘图 |
| Seaborn | 0.13.2 | 统计可视化 |
| Plotly | 6.7.0 | UI 和交互式图表 |
| XGBoost | 3.2.0 | 经典对照与扩展实现 |
| deepquantum | 4.5.0 | 光量子线路仿真框架 |
| tqdm | 4.67.3 | 进度条 |
| joblib | 1.5.3 | 模型持久化 |
| PyYAML | 6.0.3 | 配置文件解析 |
| requests | 2.33.1 | 网络请求 |
| LightGBM | 4.6.0 | 可选的经典树模型 |
| wandb | 0.25.1 | 可选实验记录 |

### 第三方依赖

| 依赖名称 | 使用版本 | 下载链接 | 安装方式 | 用途 |
|----------|----------|----------|----------|------|
| deepquantum | 4.5.0 | https://github.com/turingq/deepquantum | `pip install deepquantum` | GBS 线路构建、测量与采样 |
| XGBoost | 3.2.0 | https://xgboost.readthedocs.io/ | `conda install -c conda-forge xgboost` 或 `pip install xgboost` | 经典对照与扩展实现 |
| NetworkX | 3.4.2 | https://networkx.org/ | `conda install networkx` 或 `pip install networkx` | 图结构构建与子图抽取 |
| Matplotlib | 3.8.4 | https://matplotlib.org/ | `conda install matplotlib` | 训练曲线与图表绘制 |
| Plotly | 6.7.0 | https://plotly.com/python/ | `pip install plotly` | UI 交互图表 |
| seaborn | 0.13.2 | https://seaborn.pydata.org/ | `pip install seaborn` | 统计图辅助 |
| wandb | 0.25.1 | https://wandb.ai/ | `pip install wandb` | 可选实验记录 |

### Python 依赖安装

```bash
conda activate qgad
pip install -r Deepquantum/requirements.txt
```

如果需要重建 Conda 环境，也可以直接使用：

```bash
conda env create -f Deepquantum/environment.yml
```

`Deepquantum/environment.yml` 中锁定的主要版本如下：

| 包 | 版本 |
|----|------|
| python | 3.10.20 |
| pytorch | 2.10.0 |
| torchvision | 0.26.0 |
| numpy | 1.26.4 |
| pandas | 2.3.3 |
| scipy | 1.15.2 |
| scikit-learn | 1.7.2 |
| xgboost | 3.2.0 |
| lightgbm | 4.6.0 |
| networkx | 3.4.2 |
| matplotlib | 3.8.4 |
| seaborn | 0.13.2 |
| tqdm | 4.67.3 |
| joblib | 1.5.3 |
| pyyaml | 6.0.3 |
| requests | 2.33.1 |
| ipykernel | 7.2.0 |
| deepquantum | 4.5.0 |
| wandb | 0.25.1 |
| plotly | 6.7.0 |

## 配置说明

### 数据与模型路径

项目主要通过代码中的默认路径管理数据、缓存与结果文件。常用路径如下：

| 资源 | 默认路径 | 说明 |
|------|----------|------|
| 原始数据 | `Deepquantum/data/elliptic/raw/` | 存放 `edgelist.csv`、`features.csv`、`classes.csv` |
| 预处理缓存 | `Deepquantum/data/elliptic/processed/` | 保存量子编码参数缓存 |
| 主模型权重 | `Deepquantum/checkpoints/elliptic_model.pt` | Q-GAD 训练完成后的 checkpoint |
| 经典基线模型 | `Deepquantum/experiments/shared_models/elliptic_xgboost_model.pkl` | 共享的 XGBoost 基线 |
| 阈值评估输出 | `Deepquantum/outputs/threshold_eval/` | 保存概率文件与阈值评估结果 |
| UI 数据包 | `ui/storage/monitor_bundle.v2.json` | Streamlit 监控页面的数据来源 |

### 模型结构

Q-GAD 主模型由量子特征核和混合分类器两部分组成：

| 模块 | 作用 | 对应代码 |
|------|------|----------|
| `QuantumFeatureExtractor` | 从 GBS 采样统计中提取 9 维量子特征，并形成量子拓扑表征 | `Deepquantum/src/gbs/gbs_kernel.py` |
| `HybridNeuralClassifier` | 融合量子特征与经典特征，并输出二分类结果 | `Deepquantum/src/models/hybrid_classifier.py` |

在当前实现中，融合分类器是前馈神经网络，默认隐藏层为 `[64, 32]`，输入特征包含：

- 9 维量子特征
- 166 维经典节点特征

仓库中还保留了 `XGBoostFusionClassifier` 和 `run_xgboost.py`，分别用于后续扩展和经典方法对照。

### 训练相关参数

| 参数 | 常用值 | 说明 |
|------|--------|------|
| `max_nodes` | 20 | 每个局部子图最多保留的节点数 |
| `ego_radius` | 1.5 | Ego-network 提取半径 |
| `n_shots` | 15 / 100 / 1000 | GBS 采样次数，训练/实验会根据速度需求调整 |
| `hidden_dims` | `[64, 32]` | 混合分类器隐藏层 |
| `decision_threshold` | `0.5` / `0.26` | 二分类判定阈值 |
| `train_periods` | `(1, 34)` | 默认训练时间窗口 |
| `test_periods` | `(35, 49)` | 默认测试时间窗口 |

### 训练与评估流程

主训练脚本 [run_elliptic.py](Deepquantum/run_elliptic.py) 的处理链路是：

1. 读取 Elliptic++ 原始数据。
2. 以时间窗口 `(1, 34)` 和 `(35, 49)` 划分训练与测试集。
3. 对训练集再划分验证集。
4. 预计算每个节点的量子编码参数。
5. 构建 Q-GAD 模型并进行交替训练。
6. 在验证集上监控指标并保存 best checkpoint。
7. 在测试集上评估最终性能，并输出训练曲线与结果文件。

当前项目已有的汇总结果中，Q-GAD 的代表性指标如下：

| 指标 | 数值 |
|------|------|
| 参数量 | 38,234 |
| Accuracy | 96.63% |
| Precision | 0.9321 |
| Recall | 0.5199 |
| F1 | 0.6675 |
| AUC | 0.9256 |
| AP | 0.6924 |
| 决策阈值 | 0.26 |

## 数据集

### 数据集说明

本项目在 Elliptic++ 比特币交易数据集上开展系统性实验。该数据集包含交易图结构、节点特征和标签信息，是公开的金融黑产对抗场景基准之一。

| 数据集名称 | 来源 | 格式 | 说明 |
|-----------|------|------|------|
| Elliptic++ Bitcoin Transaction Dataset | 论文发布说明 + Kaggle 当前下载入口 | CSV | 主训练与测试数据，含图结构、特征和标签 |
| 量子编码缓存 | 本地预处理 | PKL | 保存 `squeezing`、`unitary` 与元数据 |
| 实验结果文件 | 本地生成 | JSON / CSV | 保存评估指标和对比结果 |

### 数据获取与样本

Elliptic++ 的论文摘要中注明，数据集发布于 `github.com/git-disl/EllipticPlusPlus`。  
而本项目代码当前采用的直接下载入口是 Kaggle 数据集：

- https://www.kaggle.com/datasets/ellipticdata/transaction-data

如果是按照本仓库的脚本运行，优先使用上面的 Kaggle 链接。

为了方便快速理解数据结构，项目还提供了一个小型示例样本目录：

- `Deepquantum/data/samples/`
- 说明文档：`Deepquantum/data/samples/README.md`

该示例目录已经按 Git 规则保留在仓库中，适合在不下载完整数据的情况下快速查看字段、样本规模和目录组织方式。

根据当前项目汇总文件，Elliptic++ 数据的代表性统计为：

| 统计项 | 数值 |
|--------|------|
| 总节点数 | 203,768 |
| 图中节点数 | 46,564 |
| 图中边数 | 36,624 |
| 特征维度 | 166 |
| 许可节点数 | 42,019 |
| 非法节点数 | 4,545 |
| 训练样本数 | 23,915 |
| 验证样本数 | 5,979 |
| 测试样本数 | 16,670 |

### 数据组织方式

项目默认读取以下原始文件：

- `Deepquantum/data/elliptic/raw/edgelist.csv`
- `Deepquantum/data/elliptic/raw/features.csv`
- `Deepquantum/data/elliptic/raw/classes.csv`

训练前会先做如下处理：

1. 读取原始图、特征和标签文件。
2. 过滤未知标签节点。
3. 按时间区间划分训练集和测试集。
4. 为每个节点抽取局部 ego-subgraph。
5. 将子图转换为 GBS 所需的量子编码参数。
6. 将结果缓存到 `processed/` 目录，减少重复计算。

对应的核心实现主要位于：

- `Deepquantum/src/data/elliptic_dataset.py`
- `Deepquantum/src/data/financial_dataset.py`
- `Deepquantum/src/data/temporal_graph.py`

在当前实现中，缓存的是图到量子线路的输入参数，而不是最终分类概率。  
也就是说，`squeezing` 和 `unitary` 可以预先计算，但模型前向得到的 9 维量子特征仍会在训练/评估过程中重新提取。

### 示例样本目录

`Deepquantum/data/samples/` 中的样本数据只用于说明数据格式，不用于正式训练。  
目录内包含三个最关键的文件：

- `edgelist_sample.csv`
- `features_sample.csv`
- `classes_sample.csv`

对应的 `README.md` 会解释每个字段的含义、与原始 Elliptic++ 数据的关系，以及如何用这些样本快速理解主数据加载流程。

### 目录结构示例

```text
Deepquantum/data/elliptic/
├── raw/
│   ├── edgelist.csv
│   ├── features.csv
│   └── classes.csv
└── processed/
    └── quantum_params_m20_r1.5.pkl
```

```text
Deepquantum/data/samples/
├── README.md
├── edgelist_sample.csv
├── features_sample.csv
└── classes_sample.csv
```

## 快速开始

### 训练主模型

```bash
conda activate qgad
python Deepquantum/run_elliptic.py
```

### 快速验证

```bash
python Deepquantum/run_elliptic_fast.py
```

### 经典基线

```bash
python Deepquantum/run_xgboost.py
python Deepquantum/gnn_baseline/run_gnn_baseline.py --model all --epochs 10
```

### 阈值评估

```bash
python Deepquantum/scripts/eval_threshold_full.py --device cuda --n-shots 15
```

### 生成图表

```bash
python -m Deepquantum.visualization.generate_all
```

### 构建并启动监控界面

```bash
python ui/scripts/build_monitor_bundle.py
python -m streamlit run ui/app.py
```

## 项目结构

```text
qgad-project/
├── Deepquantum/              # 主工程
│   ├── run_elliptic.py       # 主训练入口，完整训练 Q-GAD
│   ├── run_elliptic_fast.py  # 快速验证入口，适合调试和跑通流程
│   ├── run_xgboost.py        # 训练经典 XGBoost 基线
│   ├── scripts/
│   │   └── eval_threshold_full.py   # 阈值搜索与概率文件评估
│   ├── configs/              # 配置系统
│   │   ├── config.py         # ExperimentConfig 主配置类
│   │   └── __init__.py
│   ├── src/                  # 核心源码
│   │   ├── data/             # 数据读取、清洗、时序切分、PyTorch Dataset
│   │   │   ├── elliptic_dataset.py
│   │   │   ├── financial_dataset.py
│   │   │   └── temporal_graph.py
│   │   ├── gbs/              # GBS 量子核与量子特征提取
│   │   │   └── gbs_kernel.py
│   │   ├── models/           # 混合神经网络分类器
│   │   │   └── hybrid_classifier.py
│   │   └── utils/            # 图处理、XGBoost 辅助、公共工具
│   │       ├── graph_utils.py
│   │       ├── elliptic_xgb.py
│   │       ├── helpers.py
│   │       └── distributed.py
│   ├── experiments/          # 鲁棒性/机制验证实验实现与结果
│   │   ├── ablation_study/           # 消融实验
│   │   ├── adversarial_robustness/   # 对抗鲁棒性实验
│   │   ├── feature_forgery_resistance/ # 特征伪造抵抗实验
│   │   ├── feature_sensitivity/      # 特征敏感性分析
│   │   ├── quantum_protection_effect/ # 量子保护效果分析
│   │   ├── temporal_generalization/  # 时序泛化实验
│   │   └── topological_invariance/   # 拓扑不变量实验
│   ├── gnn_baseline/         # 经典 GNN 基线
│   │   ├── models/           # GCN / GAT / GraphSAGE / GIN 等
│   │   ├── utils/            # 对比分析与辅助脚本
│   │   └── run_gnn_baseline.py
│   ├── visualization/        # 图表、对比图、实验看板生成
│   │   ├── common_plots.py
│   │   ├── data_sources.py
│   │   ├── generate_all.py
│   │   └── model_comparison.py
│   ├── data/                 # Elliptic++ 数据与缓存
│   │   ├── elliptic/         # 完整数据集目录
│   │   │   ├── raw/          # 原始 CSV
│   │   │   └── processed/    # 预处理缓存
│   │   └── samples/          # 小型示例样本（已纳入 Git）
│   ├── checkpoints/          # 模型权重与训练产物
│   ├── outputs/              # 结果、曲线、评估文件与图表
│   └── requirements.txt      # Python 依赖清单
├── Deepquantum/environment.yml  # Conda 环境定义
├── Deepquantum/experiment_summary.json  # 训练与评估汇总结果
├── ui/                       # Streamlit 风险监控台与本地构建脚本
└── README.md                 # 当前完整说明文档
```

说明：各实验目录下通常包含 `README.md`、`run*.py` 和 `results/` 子目录，用于保存实验说明、执行脚本与结果文件。

### 目录职责速览

| 目录 | 主要职责 |
|------|----------|
| `Deepquantum/src/data/` | 负责 Elliptic++ 原始数据读取、时间切分、图构建与数据集封装 |
| `Deepquantum/src/gbs/` | 负责把局部子图映射成 GBS 量子编码参数并提取量子特征 |
| `Deepquantum/src/models/` | 负责将量子特征与经典特征融合后完成分类 |
| `Deepquantum/experiments/` | 负责七组实验及其结果保存 |
| `Deepquantum/gnn_baseline/` | 负责经典 GNN 对照实验，用于与 Q-GAD 比较 |
| `Deepquantum/visualization/` | 负责生成图表、对比图和汇总面板 |
| `Deepquantum/data/` | 负责保存原始数据、预处理缓存和说明性样本 |
| `ui/` | 负责离线风险监控界面的数据构建与展示 |
|

## 附注

- `Deepquantum/checkpoints/`、`Deepquantum/outputs/`、`Deepquantum/data/elliptic/` 等目录通常包含较大文件，默认不纳入版本管理。
- `Deepquantum/data/samples/` 是仓库内保留的示例样本，可在不下载完整数据的情况下快速理解数据结构。
- 如果需要完整训练数据，请先准备 Elliptic++ 原始 CSV，再运行主训练脚本或数据预处理脚本。
实验复现和可视化输出建议分别参考 `Deepquantum/experiments/` 和 `Deepquantum/visualization/` 目录。
