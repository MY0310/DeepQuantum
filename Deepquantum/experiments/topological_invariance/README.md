# 拓扑不变量实验

最后更新：2026-04-23

## 入口

- 脚本：`experiments/topological_invariance/run_with_qgad_model.py`
- 模型：`checkpoints/elliptic_model.pt`
- 后端：仅允许真实 DeepQuantum（禁用 mock）

## 运行示例

```bash
cd Deepquantum
python experiments/topological_invariance/run_with_qgad_model.py --device cuda --n-pairs 20 --n-nodes 20 --similarity-method cosine --n-shots 15
```

## 输出文件

- `experiments/topological_invariance/results/topological_invariance_qgad_results.json`
- `experiments/topological_invariance/results/topological_invariance_qgad_comparison.csv`

## 最新结果入口

- 汇总记录：`experiments/RESULTS.md`
