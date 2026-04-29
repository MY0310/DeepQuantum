# 对抗鲁棒性实验

最后更新：2026-04-24

## 入口

- 脚本：`experiments/adversarial_robustness/run_adversarial_real.py`
- 模型：`checkpoints/elliptic_model.pt`
- 后端：仅允许真实 DeepQuantum（禁用 mock）

## 运行示例

```bash
cd Deepquantum
python experiments/adversarial_robustness/run_adversarial_real.py --device cuda --max-samples 128 --batch-size 16 --num-workers 2 --decision-threshold 0.26 --epsilon 0.01 0.05 0.10 --n-shots 15 --attack-feature-ratio 0.10
```

说明：
- 当前 Windows 会话下若 `--num-workers > 0` 触发 `WinError 5`，脚本会自动回退 `num_workers=0` 并继续运行。
- 默认会对同一批样本先一次性物化量子特征并缓存到 `experiments/adversarial_robustness/cache/`，后续重跑显著提速。
- `--attack-feature-ratio` 表示每个样本被扰动的经典特征占比（按梯度绝对值 Top-k 选择）；`1.0` 为全特征攻击，`0.10` 为受限攻击者场景。

## 输出文件

- `experiments/adversarial_robustness/results/qgad_robustness_results.json`
- `experiments/adversarial_robustness/results/qgad_adversarial_results.csv`

## 最新结果入口

- 汇总记录：`experiments/RESULTS.md`
