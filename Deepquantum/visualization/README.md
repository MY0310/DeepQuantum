# Visualization

最后更新：2026-05-03

本目录只保留当前论文与答辩需要的可视化逻辑。模型对比图与安全实验图分开生成，避免不同论证线混在一起。

## 入口

```bash
conda activate qgad

# 方式 A：在 app root 执行（先进入 Deepquantum/）
cd Deepquantum

# 生成全部论文图
python -m visualization.generate_all

# 仅生成 Q-GAD vs GNN 对比图
python -m visualization.model_comparison

# 仅生成 experiments/ 实验图
python -m visualization.experiments_dashboard

# 方式 B：在 repo root 执行（不进入 Deepquantum/）
python -m Deepquantum.visualization.generate_all
python -m Deepquantum.visualization.model_comparison
python -m Deepquantum.visualization.experiments_dashboard
```

## 模块结构

```text
visualization/
├── generate_all.py          # 统一入口（调用下方两个模块）
├── model_comparison.py      # Q-GAD vs GNN 对比图
├── experiments_dashboard.py # 实验叙事图（3 张）
├── data_sources.py          # 数据读取（集中管理，不硬编码指标）
├── common_plots.py          # 通用绘图工具
└── mpl_setup.py             # matplotlib 后端与字体配置
```

## 输出

- 模型对比：`outputs/visualizations/model_comparison/`
  - `model_comparison_story.png` / `.pdf`
  - `model_comparison_metrics.csv`
- 实验图：`outputs/visualizations/experiments/`
  - `experiments_story_01_attack_pipeline.png` / `.pdf`
  - `experiments_story_02_generalization_mechanism.png` / `.pdf`
  - `experiments_story_03_evidence_constellation.png` / `.pdf`

## 图形设计

- 字体：Times New Roman 优先，缺失时自动回退 serif
- 风格：低饱和高对比配色、细网格、矢量 PDF + 高分辨率 PNG 双输出
- 稳定性：固定使用 `Agg` 后端，matplotlib 缓存放在项目本地 `.mplconfig/`

## 数据来源

- 模型对比：`outputs/` 中的 `experiment_summary.json` 与 `gnn_baseline/outputs/*/table2_metrics.csv`
- 实验图：`experiments/*/results/*.json`（按主题合并为 3 张叙事图）
- 所有数据读取集中在 `data_sources.py`，绘图脚本不硬编码指标
