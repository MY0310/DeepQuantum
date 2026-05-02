# 拓扑不变量实验

最后更新：2026-05-02

## 入口

- 脚本：`experiments/topological_invariance/run_with_qgad_model.py`
- 模型：`checkpoints/elliptic_model.pt`
- 后端：仅允许真实 DeepQuantum（禁用 mock）

## 运行示例

```bash
cd Deepquantum
conda run -n qgad python experiments/topological_invariance/run_with_qgad_model.py --device cuda --n-pairs 20 --n-nodes 20 --similarity-method cosine --feature-normalization zscore --n-shots 100 --no-canonicalize-graph
```

说明：
- `--feature-normalization` 新增支持：`none|zscore|robust`（默认 `zscore`）。
- 建议保持 `cosine + zscore`，并提高 `n_shots`（如 `100`）以提升同构均值与分离度。
- `--canonicalize-graph/--no-canonicalize-graph` 可控制编码前节点重排；当前推荐 `--no-canonicalize-graph`。

## 最新结果（2026-05-02）

- 参数：`n_pairs=20, n_nodes=20, similarity=cosine, feature_normalization=zscore, canonicalize_graph=false, n_shots=100, seed=42, device=cuda`
- 同构平均相似度：`0.6295 ± 0.3308`
- 非同构平均相似度：`0.0706 ± 0.5233`
- 分离度：`+0.5589`（`passed=true`）

## 输出文件

- `experiments/topological_invariance/results/topological_invariance_qgad_results.json`
- `experiments/topological_invariance/results/topological_invariance_qgad_comparison.csv`

## 最新结果入口

- 汇总记录：`experiments/RESULTS.md`
