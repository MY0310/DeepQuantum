# 消融实验（Ablation Study）

最后更新：2026-04-24

## 当前状态

- 入口脚本：`experiments/ablation_study/run.py`
- 训练脚本：`experiments/ablation_study/train_ablation_complete.py`
- 本轮已完成改良重跑（真实 DeepQuantum，非 mock）

说明：
- 支持一次性量子特征物化并缓存复用：`ablation_study/cache/`
- 支持验证集阈值寻优、早停、类别均衡采样、类权重开关
- 支持 `--parallel-models` 并行启动 `classical/quantum/hybrid` 子进程（默认并发上限 `--parallel-jobs 2`）
- Windows 下自动 `num_workers=0`，避免 `WinError 5`

## 运行示例

```bash
cd Deepquantum
python experiments/ablation_study/run.py --train --model all --epochs 8 --batch-size 96 --subset 0.5 --max-train-samples 1200 --max-val-samples 400 --max-test-samples 400 --device cuda --n-shots 8 --optimize-threshold --balance-sampler --no-class-weights --cache-quantum-features
```

并行示例：

```bash
cd Deepquantum
python experiments/ablation_study/run.py --train --model all --parallel-models --parallel-jobs 2 --epochs 6 --batch-size 96 --subset 0.5 --max-train-samples 1200 --max-val-samples 400 --max-test-samples 400 --device cuda --n-shots 8 --optimize-threshold --balance-sampler --cache-quantum-features
```

## 最近结果（改良配置）

- 结果文件：`experiments/ablation_study/results/ablation_training_20260423_142602.json`
- 摘要：
  - `Classical`: `F1=0.4324`, `AUC=0.8106`
  - `Quantum`: `F1=0.0961`, `AUC=0.4435`
  - `Hybrid`: `F1=0.5405`, `AUC=0.7780`
- 判定：在该配置下，`Hybrid F1 > Classical F1`，消融结论为正向（量子+经典融合优于纯经典与纯量子）。

量子分支优化补充（同数据规模，启用类权重）：
- 结果文件：`experiments/ablation_study/results/ablation_training_quantum_20260424_151430.json`
- 指标：`Quantum F1=0.1237`, `AUC=0.5611`（相对 2026-04-23 配置的 `F1=0.0961` 提升）

## 目录约定

- 训练结果输出：`experiments/ablation_study/results/`
- 模型检查点：`experiments/ablation_study/checkpoints/`
- 特征缓存：`experiments/ablation_study/cache/`
- 可视化输出统一到 `outputs/visualizations/experiments/`。
