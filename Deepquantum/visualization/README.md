# Visualization

最后更新：2026-04-24

本目录只保留当前论文与答辩需要的可视化逻辑。模型对比图与安全实验图分开生成，避免不同论证线混在一起。

## 入口

```bash
# 生成全部论文图
python -m visualization.generate_all

# 仅生成 Q-GAD vs GNN 对比图
python -m visualization.model_comparison

# 仅生成 experiments/ 实验图
python -m visualization.experiments_dashboard
```

## 输出

- 模型对比：`outputs/visualizations/model_comparison/`
  - `model_comparison_story.png`
  - `model_comparison_story.pdf`
  - `model_comparison_metrics.csv`
- 实验图：`outputs/visualizations/experiments/`
  - `experiments_story_01_attack_pipeline.png/.pdf`
  - `experiments_story_02_generalization_mechanism.png/.pdf`
  - `experiments_story_03_evidence_constellation.png/.pdf`

## 图形设计

- 字体：Times New Roman 优先，若本机缺失则自动回退到兼容 serif 字体。
- 风格：顶刊论文风格，使用低饱和高对比配色、细网格、矢量 PDF 与高分辨率 PNG 双输出。
- 稳定性：固定使用 `Agg` 后端，并将 matplotlib 缓存放在项目本地 `.mplconfig/`。

## 数据来源

- 模型对比：`experiment_summary.json` 与 `gnn_baseline/outputs/experiment_*/table2_metrics.csv`
- 实验图：`experiments/*/results/*.json`（按主题合并为 3 张叙事图）
- 数据读取集中在 `visualization/data_sources.py`，绘图脚本不再硬编码过时指标。
