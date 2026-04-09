# 📊 GNN Baseline Project Structure

Complete overview of the GNN baseline comparison framework.

## 🎯 Purpose

This directory provides **classical Graph Neural Network baselines** for fair comparison with the **quantum Gaussian Boson Sampling (GBS)** approach on financial fraud detection.

## 📂 Directory Structure

```
gnn_baseline/
│
├── models/                           # GNN model implementations
│   ├── __init__.py                   # Module exports
│   ├── gnn_models.py                 # GNN architectures (GCN, GAT, GraphSAGE, GIN)
│   │   ├── GNNConfig                 # Configuration dataclass
│   │   ├── GraphConvLayer            # GCN convolution layer
│   │   ├── MultiHeadAttentionLayer   # GAT attention mechanism
│   │   ├── GNNFeatureExtractor       # Graph → embedding extractor
│   │   └── ClassicalGNNSystem        # Complete fraud detection system
│   │
│   └── gnn_trainer.py                # Training pipeline
│       ├── GNNTrainer                # Trainer class
│       ├── train_epoch()             # Single epoch training
│       ├── evaluate()                # Validation/testing
│       └── train()                   # Full training loop
│
├── utils/                            # Analysis utilities
│   └── comparison_analysis.py        # Quantum vs classical comparison
│       ├── QuantumClassicalComparator
│       ├── compare_metrics()         # Metrics table
│       ├── plot_metric_comparison()  # Visualizations
│       ├── analyze_training_dynamics()
│       └── generate_summary_report()
│
├── outputs/                          # Generated experiment results
│   └── experiment_YYYYMMDD_HHMMSS/
│       ├── gcn/
│       │   ├── gcn_best_model.pt
│       │   ├── gcn_history.json
│       │   ├── gcn_test_metrics.json
│       │   └── gcn_training_curves.png
│       ├── gat/
│       ├── sage/
│       ├── gin/
│       ├── experiment_summary.json
│       └── metrics_comparison.csv
│
├── analysis/                         # Comparison outputs (auto-generated)
│   ├── metrics_comparison.csv
│   ├── metrics_comparison.png
│   ├── training_dynamics.png
│   ├── comparison_report.txt
│   └── comparison_report.md
│
├── checkpoints/                      # Model checkpoints (auto-generated)
│   ├── gcn_checkpoints/
│   ├── gat_checkpoints/
│   └── ...
│
├── logs/                             # Training logs (auto-generated)
│   ├── gcn_logs/
│   ├── gat_logs/
│   └── ...
│
├── run_gnn_baseline.py               # Main experiment script
├── example_usage.py                  # Quick start workflow
├── README.md                         # User documentation
└── PROJECT_STRUCTURE.md              # This file
```

## 🔧 Core Components

### 1. GNN Models (`models/gnn_models.py`)

#### Architectures Implemented

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `GNNConfig` | Configuration | Dataclass for hyperparameters |
| `GraphConvLayer` | GCN convolution | `forward(x, edge_index)` |
| `MultiHeadAttentionLayer` | GAT attention | `forward(x, edge_index)` |
| `GNNFeatureExtractor` | Graph → embedding | `forward(x, edge_index, batch)` |
| `ClassicalGNNSystem` | Complete system | `forward()`, `predict()` |

#### Usage Example

```python
from models import GNNConfig, create_gnn_model

# Create configuration
config = GNNConfig(
    gnn_type='gcn',
    num_layers=3,
    hidden_dim=64,
    dropout=0.1
)

# Create model
model = create_gnn_model('gcn', config)

# Forward pass
logits = model(x, edge_index, batch, classical_features)
```

### 2. GNN Trainer (`models/gnn_trainer.py`)

#### GNNTrainer Class

```python
trainer = GNNTrainer(
    model=model,
    device='cuda',
    learning_rate=1e-3
)

# Training
trainer.train(
    train_loader,
    val_loader,
    n_epochs=10,
    early_stopping_patience=5
)

# Evaluation
test_metrics = trainer.evaluate(test_loader)
```

#### Training History

```python
trainer.history = {
    'train_loss': [...],
    'train_acc': [...],
    'val_loss': [...],
    'val_acc': [...],
    'val_f1': [...],
    'val_auc': [...],
    'val_precision': [...],
    'val_recall': [...]
}
```

### 3. Comparison Analysis (`utils/comparison_analysis.py`)

#### QuantumClassicalComparator

```python
comparator = QuantumClassicalComparator(
    quantum_results_dir='../outputs/elliptic_fast_test',
    classical_results_dir='./gnn_baseline/outputs'
)

# Compare metrics
df = comparator.compare_metrics()

# Generate plots
comparator.plot_metric_comparison(df)
comparator.analyze_training_dynamics()

# Generate report
comparator.generate_summary_report()
```

## 🚀 Workflow

### Step 1: Train GNN Baselines

```bash
# Single model
python run_gnn_baseline.py --model gcn --epochs 10

# All models
python run_gnn_baseline.py --model all --epochs 20

# Fast testing
python run_gnn_baseline.py --model gcn --epochs 5 --fast
```

### Step 2: Run Comparison Analysis

```bash
python utils/comparison_analysis.py \
    --quantum-dir ../outputs/elliptic_fast_test \
    --classical-dir ./gnn_baseline/outputs
```

### Step 3: Review Results

```bash
# View comparison report
cat gnn_baseline/analysis/comparison_report.txt

# Open visualization
start gnn_baseline/analysis/metrics_comparison.png
```

## 📊 Data Flow

```
Elliptic++ Dataset
    ↓
FinancialGraphDataset
    ↓
DataLoader (batches)
    ↓
    ├─→ Graph Data (adj, node_features)
    │   ↓
    │   GNNFeatureExtractor
    │   ↓
    │   Graph Embeddings [batch, 9]
    │
    └─→ Classical Features
        ↓
        ClassicalEncoder
        ↓
        Tabular Embeddings [batch, 32]
    ↓
Fusion Layer (concat + MLP)
    ↓
Classifier (binary)
    ↓
Fraud Predictions
```

## 🔬 Experimental Design

### Controlled Variables

Both quantum and classical systems use:

| Parameter | Value | Reason |
|-----------|-------|--------|
| Dataset | Elliptic++ | Bitcoin transaction fraud |
| Train/Val/Test Split | 80/20, periods 1-34 / 35-49 | Temporal split |
| Batch Size | 32 | Standard for fraud detection |
| Learning Rate | 1e-3 | Adam default |
| Optimizer | Adam | Robust for GNNs |
| Dropout | 0.1 | Prevent overfitting |
| Graph Feature Dim | 9 | Match quantum output |
| Classical Feature Dim | 10 | Tabular features |

### Evaluated Metrics

| Metric | Formula | Importance |
|--------|---------|------------|
| **AUC-ROC** | $\int TPR(FPR^{-1})$ | Ranking quality |
| **F1 Score** | Harmonic mean of P, R | Precision-recall balance |
| **Precision** | $TP/(TP+FP)$ | False alarm rate |
| **Recall** | $TP/(TP+FN)$ | Fraud detection rate |

## 📈 Expected Results

Based on literature and pilot experiments:

| Model | AUC | Training Time | Inference Speed |
|-------|-----|---------------|-----------------|
| **GBS Quantum** | 0.90-0.94 | Very Slow (hours) | Slow |
| **GCN** | 0.86-0.90 | Fast (minutes) | Fast |
| **GAT** | 0.87-0.91 | Medium | Medium |
| **GraphSAGE** | 0.86-0.90 | Fast | Fast |
| **GIN** | 0.87-0.91 | Medium | Medium |

## 🛠️ Customization

### Adding New GNN Architectures

Edit `models/gnn_models.py`:

```python
class YourGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        # Define your layer
        pass

    def forward(self, x, edge_index):
        # Implement message passing
        pass

# Register in GNNFeatureExtractor
elif self.gnn_type == "your_gnn":
    self.convs.append(YourGNNLayer(...))
```

### Adding New Metrics

Edit `models/gnn_trainer.py` in `evaluate()`:

```python
from sklearn.metrics import your_metric

metrics['your_metric'] = your_metric(all_labels, all_preds)
```

### Custom Visualization

Edit `utils/comparison_analysis.py`:

```python
def plot_custom_comparison(self):
    fig, ax = plt.subplots()
    # Your plotting code
    plt.savefig(self.output_dir / "custom_plot.png")
```

## 📚 References

### Papers

1. **GCN**: Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017
2. **GAT**: Veličković et al., "Graph Attention Networks", ICLR 2018
3. **GraphSAGE**: Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017
4. **GIN**: Xu et al., "How Powerful are Graph Neural Networks?", ICLR 2019

### Code Libraries

- PyTorch: https://pytorch.org/
- PyTorch Geometric: https://pyg.org/
- scikit-learn: https://scikit-learn.org/

## 🤝 Contributing

To add features or fix bugs:

1. Modify code in respective directories
2. Test with `--fast` flag first
3. Update this document if structure changes
4. Run full comparison suite before committing

---

**Version**: 1.0.0
**Last Updated**: 2025-01-12
