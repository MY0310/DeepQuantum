# Elliptic++ 示例样本

这个目录提供了一组可直接阅读的微型样本，用来帮助理解 Elliptic++ 数据在本项目中的组织方式。

它们不是从完整数据集中裁剪出来的正式训练片段，而是用来说明字段、目录结构和加载顺序的说明性文件。

## 与代码的对应关系

项目里的数据加载逻辑主要在以下文件中：

| 代码文件 | 作用 |
|----------|------|
| `Deepquantum/src/data/elliptic_dataset.py` | 读取 Elliptic++ 原始 CSV、处理标签、构建图、按时间切分 |
| `Deepquantum/src/data/financial_dataset.py` | 把图、特征、标签包装成 PyTorch Dataset，并预计算量子参数 |
| `Deepquantum/src/data/temporal_graph.py` | 处理带时间戳的动态图快照和时间窗口 |

在这些代码里，默认会读取：

- `edgelist.csv`
- `features.csv`
- `classes.csv`

而 `Deepquantum/data/samples/` 则提供同类结构的缩小版样例，帮助快速理解这些文件分别扮演什么角色。

## 文件说明

| 文件 | 作用 | 对应原始文件 |
|------|------|--------------|
| `edgelist_sample.csv` | 交易图边信息 | `edgelist.csv` |
| `features_sample.csv` | 节点特征 | `features.csv` |
| `classes_sample.csv` | 节点标签 | `classes.csv` |

## 字段说明

### `edgelist_sample.csv`

这类文件描述交易图的边。

- `txId1`：源节点
- `txId2`：目标节点
- `time`：时间窗口
- `weight`：边权重

在真实数据里，时间列会参与时序划分，图也会被进一步用于提取局部 ego-subgraph。

### `features_sample.csv`

这类文件描述节点特征。

- `txId`：节点 ID
- `time`：节点所属时间窗口
- `feature_1` - `feature_5`：示例特征列

真实的 Elliptic++ 特征文件一共有 166 维，这里为了便于阅读，只保留了少量特征列。

### `classes_sample.csv`

这类文件描述节点标签。

- `txId`：节点 ID
- `class`：类别标签
  - `0` 表示 unknown
  - `1` 表示 illicit / fraud
  - `2` 表示 licit / normal

在项目代码里，训练时会把它进一步转成二分类目标：

- `1` -> `1`，表示欺诈
- `2` -> `0`，表示正常

`unknown` 节点会在预处理阶段被过滤掉。

## 文件如何被读取

按照 `Deepquantum/src/data/elliptic_dataset.py` 的默认逻辑：

1. 先读取 `edgelist.csv`
2. 再读取 `features.csv`，并把第一列作为索引
3. 最后读取 `classes.csv`，并把第一列作为索引
4. 过滤掉未知标签节点
5. 保留同时出现在特征和标签中的节点
6. 按时间区间划分训练集与测试集

这个流程也是为什么项目里会同时存在“原始数据、缓存数据、示例数据”三类目录。

## 使用说明

这份样本数据只用于快速理解数据格式，不用于正式训练。

如果你想跑完整训练，需要下载 Elliptic++ 原始数据集，并把以下文件放入：

`Deepquantum/data/elliptic/raw/`

- `edgelist.csv`
- `features.csv`
- `classes.csv`

然后再运行主训练脚本或数据加载脚本。
