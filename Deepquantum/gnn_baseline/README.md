# GNN Baseline for Q-GAD Comparison

Classical Graph Neural Network baselines for fair comparison with quantum Gaussian Boson Sampling (GBS) approach on the Elliptic++ Bitcoin fraud detection dataset.

## 📁 Directory Structure

```
gnn_baseline/
├── models/
│   ├── __init__.py
│   ├── gnn_models.py          # GNN architectures (GCN, GAT, GraphSAGE, GIN)
│   └── gnn_trainer.py         # Training pipeline
├── utils/
│   └── comparison_analysis.py # Quantum vs classical comparison tools
├── outputs/                   # Experiment results (auto-generated)
├── run_gnn_baseline.py        # Main experiment script
└── README.md                  # This file
```

## 🎯 Supported GNN Architectures

| Architecture | Full Name | Key Feature | Paper |
|--------------|-----------|-------------|-------|
| **GCN** | Graph Convolutional Network | Spectral convolution | Kipf & Welling, ICLR 2017 |
| **GAT** | Graph Attention Network | Attention-based aggregation | Veličković et al., ICLR 2018 |
| **GraphSAGE** | SAmple and aggreGatE | Inductive learning | Hamilton et al., NeurIPS 2017 |
| **GIN** | Graph Isomorphism Network | Powerful for graph classification | Xu et al., ICLR 2019 |

## 🚀 Quick Start

### 1. Basic Usage

Train a single GNN model (e.g., GCN):

```bash
python run_gnn_baseline.py --model gcn --epochs 10
```

### 2. Fast Mode (Reduced Dataset)

Quick testing with reduced dataset:

```bash
python run_gnn_baseline.py --model gcn --epochs 5 --fast
```

### 3. Train All Architectures

Compare all GNN types:

```bash
python run_gnn_baseline.py --model all --epochs 20
```

### 4. Custom Configuration

```bash
python run_gnn_baseline.py \
    --model gat \
    --epochs 15 \
    --batch-size 64 \
    --lr 5e-4 \
    --hidden-dim 128 \
    --num-layers 4 \
    --device cuda
```

## 📊 Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `gcn` | GNN architecture: `gcn`, `gat`, `sage`, `gin`, `all` |
| `--epochs` | int | `10` | Number of training epochs |
| `--batch-size` | int | `32` | Batch size for training |
| `--lr` | float | `1e-3` | Learning rate |
| `--hidden-dim` | int | `64` | Hidden dimension for GNN layers |
| `--num-layers` | int | `3` | Number of GNN layers |
| `--fast` | flag | `False` | Enable fast mode (reduced dataset) |
| `--device` | str | `auto` | Device to use (`cuda`/`cpu`) |
| `--seed` | int | `42` | Random seed for reproducibility |

## 📈 Outputs

After running experiments, you'll find:

```
outputs/experiment_YYYYMMDD_HHMMSS/
├── gcn/
│   ├── gcn_best_model.pt              # Best model checkpoint
│   ├── gcn_history.json               # Training history
│   ├── gcn_test_metrics.json          # Test set metrics
│   └── gcn_training_curves.png        # Training curves plot
├── gat/
│   └── ...
├── experiment_summary.json            # Combined results summary
└── metrics_comparison.csv             # Metrics comparison table
```

## 📊 Comparison with Quantum Model

After running both quantum and classical experiments, generate comparison report:

```bash
python utils/comparison_analysis.py \
    --quantum-dir ../outputs/elliptic_fast_test \
    --classical-dir ./gnn_baseline/outputs \
    --output-dir ./gnn_baseline/analysis
```

This generates:

- **metrics_comparison.csv**: Numerical comparison table
- **metrics_comparison.png**: Bar chart visualization
- **training_dynamics.png**: Training curve comparison
- **comparison_report.txt**: Detailed textual report
- **comparison_report.md**: Markdown report for papers

## 🔬 Experimental Protocol for Fair Comparison

### Data Splitting

Both quantum and classical models use:
- **Train**: Periods 1-34 (80% of train data)
- **Validation**: Periods 1-34 (20% of train data)
- **Test**: Periods 35-49 (completely held-out)

### Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **AUC-ROC** | $\int TPR(FPR^{-1}) dFPR$ | Overall ranking quality |
| **F1 Score** | $2 \cdot \frac{P \cdot R}{P + R}$ | Precision-recall balance |
| **Precision** | $TP / (TP + FP)$ | False positive rate |
| **Recall** | $TP / (TP + FN)$ | Fraud detection rate |

### Hyperparameter Matching

| Parameter | Quantum (GBS) | Classical (GNN) |
|-----------|---------------|-----------------|
| Graph feature dim | 9 | 9 |
| Classical feature dim | 10 | 10 |
| Hidden dims | [64, 32] | [64, 32] |
| Dropout | 0.1 | 0.1 |
| Optimizer | Adam | Adam |
| Learning rate | 1e-3 | 1e-3 |
| Batch size | 32 | 32 |

## 🧪 Understanding the Differences

### Theoretical Differences

| Aspect | Quantum GBS | Classical GNN |
|--------|-------------|---------------|
| **Feature extraction** | Quantum sampling in Hilbert space | Message passing on graph |
| **Computational complexity** | $O(n^3 2^n)$ (Hafnian) | $O(\|E\|)$ (linear in edges) |
| **Non-linearity source** | Quantum interference | Activation functions (ReLU) |
| **Training** | Alternating (quantum + classical) | End-to-end backprop |
| **Interpretability** | Physical (photons, squeezing) | Graph-theoretic (embeddings) |

### Expected Performance

Based on literature:

- **Quantum (GBS)**: AUC ≈ 0.90-0.94
  - Pros: Theoretical advantage for topology
  - Cons: Computationally expensive, hard to train

- **Classical (GNN)**: AUC ≈ 0.88-0.92
  - Pros: Fast, scalable, mature tooling
  - Cons: May miss quantum-specific features

## 📝 Citation

If you use this baseline in your research, please cite:

```bibtex
@misc{gnn_baseline_qgad,
  title={Classical Baselines for Quantum Graph Anomaly Detection},
  author={Your Name},
  year={2025},
  note={Comparison with GBS-based Q-GAD system}
}
```

## 🛠️ Troubleshooting

### Issue: Out of Memory

```bash
# Reduce batch size
python run_gnn_baseline.py --model gcn --batch-size 16

# Or reduce hidden dimension
python run_gnn_baseline.py --model gcn --hidden-dim 32
```

### Issue: Poor Convergence

```bash
# Reduce learning rate
python run_gnn_baseline.py --model gcn --lr 5e-4

# Increase model capacity
python run_gnn_baseline.py --model gcn --hidden-dim 128 --num-layers 4
```

### Issue: Data Loading Errors

Ensure the Elliptic++ dataset is properly downloaded and preprocessed:

```bash
cd ../
python -m data.financial_dataset  # This will trigger download
```

## 📧 Contact

For questions or issues, please open an issue on the main repository.

---

**Last Updated**: 2025-01-12
