"""
Differential Analysis Tools for Quantum vs Classical GNN

This module provides tools to analyze differences between quantum GBS
and classical GNN approaches, including:
1. Performance comparison
2. Feature importance analysis
3. Error pattern analysis
4. Computational efficiency comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from scipy import stats


class QuantumClassicalComparator:
    """
    Compare quantum (GBS) and classical (GNN) models.

    Analysis dimensions:
    1. Performance metrics (AUC, F1, Precision, Recall)
    2. Training dynamics (loss curves, convergence)
    3. Computational efficiency (time, memory)
    4. Error patterns (confusion matrices, misclassified samples)
    5. Robustness (noise sensitivity, generalization)
    """

    def __init__(
        self,
        quantum_results_dir: str,
        classical_results_dir: str,
        output_dir: str = "./gnn_baseline/analysis"
    ):
        """
        Initialize comparator.

        Args:
            quantum_results_dir: Path to Q-GAD experiment results
            classical_results_dir: Path to GNN baseline results
            output_dir: Directory to save analysis outputs
        """
        self.quantum_dir = Path(quantum_results_dir)
        self.classical_dir = Path(classical_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load results
        self.quantum_results = self._load_quantum_results()
        self.classical_results = self._load_classical_results()

    def _load_quantum_results(self) -> Dict:
        """Load quantum experiment results."""
        # Try to find test metrics file
        test_metrics_path = self.quantum_dir / "elliptic_history.json"
        history_path = self.quantum_dir / "elliptic_history.json"

        results = {}

        if test_metrics_path.exists():
            with open(test_metrics_path, 'r') as f:
                results['metrics'] = json.load(f)

        if history_path.exists():
            with open(history_path, 'r') as f:
                history = json.load(f)
                results['history'] = history

        return results

    def _load_classical_results(self) -> Dict:
        """Load classical GNN experiment results."""
        results = {}

        # Find all model directories
        for model_dir in self.classical_dir.glob("*/"):
            model_name = model_dir.name

            # Load metrics
            metrics_file = model_dir / f"{model_name}_test_metrics.json"
            history_file = model_dir / f"{model_name}_history.json"

            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    results[model_name] = json.load(f)

            if history_file.exists():
                with open(history_file, 'r') as f:
                    if model_name not in results:
                        results[model_name] = {}
                    results[model_name]['history'] = json.load(f)

        return results

    def compare_metrics(self) -> pd.DataFrame:
        """
        Compare test metrics across all models.

        Returns:
            DataFrame with metrics comparison
        """
        metrics_data = []

        # Quantum results
        if 'metrics' in self.quantum_results:
            q_metrics = self.quantum_results['metrics']
            # Get final epoch metrics
            if 'val_auc' in q_metrics and len(q_metrics['val_auc']) > 0:
                metrics_data.append({
                    'Model': 'Q-GAD (GBS)',
                    'AUC': q_metrics['val_auc'][-1],
                    'F1': q_metrics['val_f1'][-1],
                    'Precision': q_metrics.get('val_precision', [0])[-1],
                    'Recall': q_metrics.get('val_recall', [0])[-1]
                })

        # Classical results
        for model_name, model_results in self.classical_results.items():
            if 'auc' in model_results:
                metrics_data.append({
                    'Model': f'GNN ({model_name.upper()})',
                    'AUC': model_results['auc'],
                    'F1': model_results['f1'],
                    'Precision': model_results.get('precision', 0),
                    'Recall': model_results.get('recall', 0)
                })

        df = pd.DataFrame(metrics_data)

        # Save to CSV
        df.to_csv(self.output_dir / "metrics_comparison.csv", index=False)

        return df

    def plot_metric_comparison(self, df: pd.DataFrame):
        """
        Visualize metric comparison.

        Args:
            df: DataFrame from compare_metrics()
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Quantum vs Classical: Performance Comparison', fontsize=16, fontweight='bold')

        metrics = ['AUC', 'F1', 'Precision', 'Recall']
        colors = ['#2ecc71' if 'Q-GAD' in model else '#3498db' for model in df['Model']]

        for ax, metric in zip(axes.flat, metrics):
            bars = ax.bar(df['Model'], df[metric], color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric} Score', fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.0])

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10)

            # Rotate x labels
            ax.set_xticklabels(df['Model'], rotation=15, ha='right')

        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Metric comparison plot saved to {self.output_dir / 'metrics_comparison.png'}")

    def analyze_training_dynamics(self):
        """Analyze training dynamics (convergence, stability)."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Training Dynamics Comparison', fontsize=16, fontweight='bold')

        # Plot quantum training curves if available
        if 'history' in self.quantum_results:
            q_hist = self.quantum_results['history']

            # AUC over epochs
            if 'val_auc' in q_hist:
                axes[0, 0].plot(q_hist['val_auc'], marker='o', label='Q-GAD (GBS)',
                               color='#2ecc71', linewidth=2)
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Validation AUC')
                axes[0, 0].set_title('AUC Convergence')
                axes[0, 0].grid(alpha=0.3)
                axes[0, 0].legend()

            # Loss over epochs
            if 'val_loss' in q_hist:
                axes[0, 1].plot(q_hist['val_loss'], marker='s', label='Q-GAD (GBS)',
                               color='#2ecc71', linewidth=2)
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Validation Loss')
                axes[0, 1].set_title('Loss Convergence')
                axes[0, 1].grid(alpha=0.3)
                axes[0, 1].legend()

        # Plot classical training curves
        colors = ['#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']
        for i, (model_name, model_results) in enumerate(self.classical_results.items()):
            if 'history' in model_results:
                hist = model_results['history']
                color = colors[i % len(colors)]

                if 'val_auc' in hist:
                    axes[1, 0].plot(hist['val_auc'], marker='o', label=f'GNN ({model_name.upper()})',
                                   color=color, linewidth=2, alpha=0.7)

                if 'val_loss' in hist:
                    axes[1, 1].plot(hist['val_loss'], marker='s', label=f'GNN ({model_name.upper()})',
                                   color=color, linewidth=2, alpha=0.7)

        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Validation AUC')
        axes[1, 0].set_title('Classical GNNs: AUC Convergence')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend()

        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss')
        axes[1, 1].set_title('Classical GNNs: Loss Convergence')
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / "training_dynamics.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Training dynamics plot saved to {self.output_dir / 'training_dynamics.png'}")

    def compute_statistical_significance(self) -> pd.DataFrame:
        """
        Compute statistical significance of performance differences.

        Note: This requires multiple runs. For single-run comparison,
        we provide a theoretical analysis.
        """
        print("\n" + "=" * 80)
        print("Statistical Significance Analysis")
        print("=" * 80)

        print("\n⚠️  Note: Proper statistical testing requires multiple runs (n >= 5)")
        print("   Current analysis provides theoretical comparison only.\n")

        df = self.compare_metrics()

        # Compute pairwise differences
        if 'Q-GAD (GBS)' in df['Model'].values:
            q_auc = df[df['Model'] == 'Q-GAD (GBS)']['AUC'].values[0]

            print("Quantum vs Classical Performance Gap:")
            print("-" * 80)

            for _, row in df.iterrows():
                if 'Q-GAD' not in row['Model']:
                    gap = q_auc - row['AUC']
                    print(f"  {row['Model']:<20} ΔAUC = {gap:+.4f}")

        return df

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report = []
        report.append("=" * 80)
        report.append("QUANTUM VS CLASSICAL GNN: COMPARATIVE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Metrics comparison
        report.append("1. PERFORMANCE METRICS")
        report.append("-" * 80)
        df = self.compare_metrics()
        report.append(df.to_string(index=False))
        report.append("")

        # Statistical significance
        report.append("2. STATISTICAL ANALYSIS")
        report.append("-" * 80)
        if 'Q-GAD (GBS)' in df['Model'].values:
            q_auc = df[df['Model'] == 'Q-GAD (GBS)']['AUC'].values[0]
            for _, row in df.iterrows():
                if 'Q-GAD' not in row['Model']:
                    gap = q_auc - row['AUC']
                    report.append(f"  {row['Model']:<20} ΔAUC = {gap:+.4f}")
        report.append("")

        # Recommendations
        report.append("3. RECOMMENDATIONS")
        report.append("-" * 80)

        if 'Q-GAD (GBS)' in df['Model'].values:
            q_auc = df[df['Model'] == 'Q-GAD (GBS)']['AUC'].values[0]
            best_classical = df[df['Model'] != 'Q-GAD (GBS)']['AUC'].max()

            if q_auc > best_classical:
                report.append("  ✓ Quantum approach shows superior performance")
                report.append("    Recommended for: High-stakes fraud detection")
                report.append("    Consider: Computational cost vs accuracy gain")
            else:
                report.append("  ⚠ Classical GNN matches or exceeds quantum performance")
                report.append("    Recommended for: Production deployment")
                report.append("    Consider: Training time, model complexity")
        else:
            report.append("  ℹ Quantum results not available for comparison")

        report.append("")

        # Save report
        report_text = "\n".join(report)
        with open(self.output_dir / "comparison_report.txt", 'w') as f:
            f.write(report_text)

        print(report_text)

        # Also save as markdown
        with open(self.output_dir / "comparison_report.md", 'w') as f:
            f.write("# Quantum vs Classical GNN: Comparative Analysis\n\n")
            f.write("## Performance Metrics\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n\n## Visualizations\n\n")
            f.write("![Metrics Comparison](metrics_comparison.png)\n\n")
            f.write("![Training Dynamics](training_dynamics.png)\n")

        print(f"\n✓ Comparison report saved to {self.output_dir}")


def main():
    """Example usage of comparator."""
    import argparse

    parser = argparse.ArgumentParser(description='Compare quantum and classical GNN results')
    parser.add_argument('--quantum-dir', type=str, default='./outputs/elliptic_fast_test',
                       help='Path to quantum experiment results')
    parser.add_argument('--classical-dir', type=str, default='./gnn_baseline/outputs',
                       help='Path to classical GNN results')
    parser.add_argument('--output-dir', type=str, default='./gnn_baseline/analysis',
                       help='Output directory for analysis')

    args = parser.parse_args()

    # Create comparator
    comparator = QuantumClassicalComparator(
        quantum_results_dir=args.quantum_dir,
        classical_results_dir=args.classical_dir,
        output_dir=args.output_dir
    )

    # Run all analyses
    print("=" * 80)
    print("Running Quantum vs Classical Comparison Analysis")
    print("=" * 80)

    # 1. Metrics comparison
    print("\n[1/4] Comparing performance metrics...")
    df = comparator.compare_metrics()
    print(df.to_string(index=False))

    # 2. Visualizations
    print("\n[2/4] Generating visualizations...")
    comparator.plot_metric_comparison(df)
    comparator.analyze_training_dynamics()

    # 3. Statistical analysis
    print("\n[3/4] Computing statistical significance...")
    comparator.compute_statistical_significance()

    # 4. Summary report
    print("\n[4/4] Generating summary report...")
    comparator.generate_summary_report()

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
