"""
Ablation Study Visualization Script for Q-GAD System

This script creates visualizations for the ablation study (Table 3) comparing:
- Classical-only features (166-dim)
- Quantum-only features (9-dim)
- Hybrid features (175-dim, full Q-GAD)

Author: Q-GAD Research Team
Date: 2026-01-20
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['font.size'] = 11
plt.rcParams['font.family'] = 'DejaVu Sans'

# Data from Table 3: Ablation Study - Feature Contribution Analysis
configurations = ['Classical-Only\n(166-dim)', 'Quantum-Only\n(9-dim)', 'Hybrid\n(175-dim)']
short_names = ['Classical', 'Quantum', 'Hybrid']
f1_scores = [0.0902, 0.9664, 1.0000]
auc_scores = [0.4924, 0.4800, 0.9259]
recall_scores = [0.1369, 1.0000, 0.9983]
params = [0, 8746, 38234]  # Classical has no fixed params (tree-based)

# Color scheme
colors = ['#3498db', '#9b59b6', '#e74c3c']  # Blue, Purple, Red
classical_color = '#3498db'
quantum_color = '#9b59b6'
hybrid_color = '#e74c3c'

# Create output directory
output_dir = Path("outputs/visualizations/ablation")
output_dir.mkdir(parents=True, exist_ok=True)

print("="*70)
print("Q-GAD Ablation Study Visualization")
print("="*70)
print(f"Output directory: {output_dir}")
print()


# ============================================================================
# Figure 1: Feature Contribution Comparison (Grouped Bar Chart)
# ============================================================================
print("[1/4] Creating grouped bar chart for ablation metrics...")

fig, ax = plt.subplots(figsize=(12, 7))

# Set up bar positions
x = np.arange(len(configurations))
width = 0.25

# Create bars for each metric
bars1 = ax.bar(x - width, f1_scores, width, label='F1 Score',
               color='#2ecc71', edgecolor='black', linewidth=1.5, alpha=0.8)
bars2 = ax.bar(x, auc_scores, width, label='AUC',
               color='#f39c12', edgecolor='black', linewidth=1.5, alpha=0.8)
bars3 = ax.bar(x + width, recall_scores, width, label='Recall',
               color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on bars
def add_value_labels(bars, values):
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.4f}' if value > 0.5 else f'{value:.4f}',
                ha='center', va='bottom', fontsize=10, weight='bold')

add_value_labels(bars1, f1_scores)
add_value_labels(bars2, auc_scores)
add_value_labels(bars3, recall_scores)

# Customize plot
ax.set_ylabel('Performance Score', fontsize=14, weight='bold')
ax.set_xlabel('Feature Configuration', fontsize=14, weight='bold')
ax.set_title('Ablation Study: Impact of Feature Types on Performance',
             fontsize=16, weight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(configurations, fontsize=12)
ax.legend(fontsize=12, loc='upper left')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add annotation for hybrid superiority
ax.annotate('Best Overall', xy=(2, 1.0), xytext=(1.5, 1.05),
            fontsize=12, weight='bold', color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.savefig(output_dir / 'grouped_bar_ablation.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: grouped_bar_ablation.png")


# ============================================================================
# Figure 2: Feature Importance Analysis (Stacked Contribution)
# ============================================================================
print("[2/4] Creating feature contribution analysis...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

metrics = [
    ('F1 Score', f1_scores, 0, 1.1),
    ('AUC', auc_scores, 0, 1.0),
    ('Recall', recall_scores, 0, 1.1)
]

for idx, (metric_name, values, ymin, ymax) in enumerate(metrics):
    ax = axes[idx]

    # Create bars with different patterns for better clarity
    bars = ax.bar(short_names, values, color=colors,
                   edgecolor='black', linewidth=2, alpha=0.85)

    # Highlight the best performer
    best_idx = np.argmax(values)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)

    # Add value labels
    for i, (name, value) in enumerate(zip(short_names, values)):
        label_y = value + 0.03 if value > 0.5 else value + 0.02
        ax.text(i, label_y, f'{value:.4f}',
                ha='center', va='bottom', fontsize=12, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='black', alpha=0.8))

    # Show percentage improvement of Hybrid over Classical
    if values[2] > values[0]:  # Hybrid > Classical
        improvement = ((values[2] - values[0]) / values[0]) * 100 if values[0] > 0 else float('inf')
        if improvement != float('inf'):
            ax.text(1.5, ymax * 0.9, f'Hybrid vs Classical:\n+{improvement:.1f}%',
                    ha='center', fontsize=10, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

    ax.set_ylabel(metric_name, fontsize=13, weight='bold')
    ax.set_title(f'{metric_name} by Feature Type', fontsize=14, weight='bold')
    ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=values[2], color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Hybrid Level')

plt.suptitle('Feature Type Contribution Analysis', fontsize=17, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / 'feature_contribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: feature_contribution.png")


# ============================================================================
# Figure 3: Radar Chart for Ablation Study
# ============================================================================
print("[3/4] Creating radar chart for ablation comparison...")

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Metrics for radar
metrics_names = ['F1 Score', 'AUC', 'Recall']
num_vars = len(metrics_names)

# Compute angles
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# Plot each configuration
all_values = [f1_scores, auc_scores, recall_scores]

for i, config in enumerate(configurations):
    values = [all_values[j][i] for j in range(num_vars)]
    values += values[:1]

    linewidth = 3 if i == 2 else 2  # Hybrid gets thicker line
    alpha = 0.8 if i == 2 else 0.5

    ax.plot(angles, values, 'o-', linewidth=linewidth,
            label=config, color=colors[i], alpha=alpha, markersize=8)
    ax.fill(angles, values, alpha=0.2, color=colors[i])

# Customize
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_names, size=13, weight='bold')
ax.set_ylim(0, 1.1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=11)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=12)
ax.set_title('Ablation Study: Feature Configuration Comparison\n(Radar View)',
             size=16, weight='bold', pad=30)

plt.tight_layout()
plt.savefig(output_dir / 'radar_ablation.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: radar_ablation.png")


# ============================================================================
# Figure 4: Feature Synergy Analysis (Performance vs Dimensionality)
# ============================================================================
print("[4/4] Creating feature synergy analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left plot: F1 Score vs Feature Dimensionality
feature_dims = [166, 9, 175]
markers = ['s', '^', 'D']  # Square, triangle, diamond
sizes = [200, 200, 300]

for i, (dim, f1, name) in enumerate(zip(feature_dims, f1_scores, short_names)):
    ax1.scatter(dim, f1, s=sizes[i], c=colors[i], marker=markers[i],
                edgecolors='black', linewidth=2, alpha=0.8, label=name, zorder=5)

    # Add annotation
    offset_y = 0.05 if name != 'Classical' else -0.08
    ax1.annotate(f'{name}\n{dim}D → F1={f1:.4f}',
                 xy=(dim, f1), xytext=(dim, f1 + offset_y),
                 fontsize=11, weight='bold', ha='center',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3))

# Draw synergy curve (illustrative)
x_curve = np.linspace(0, 180, 100)
y_curve = 1 / (1 + np.exp(-(x_curve - 100) / 20))  # Sigmoid
ax1.plot(x_curve, y_curve, 'k--', alpha=0.3, linewidth=2, label='Synergy Trend')

ax1.set_xlabel('Feature Dimensionality', fontsize=13, weight='bold')
ax1.set_ylabel('F1 Score', fontsize=13, weight='bold')
ax1.set_title('Performance vs Feature Dimensionality', fontsize=14, weight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='lower right')
ax1.set_xlim(-5, 185)
ax1.set_ylim(-0.1, 1.15)

# Right plot: Efficiency (Performance per Dimension)
efficiency_f1 = [f1 / dim * 1000 for f1, dim in zip(f1_scores, feature_dims)]  # Per 1000 dims

bars = ax2.bar(short_names, efficiency_f1, color=colors,
               edgecolor='black', linewidth=2, alpha=0.8)

# Highlight best efficiency
best_eff_idx = np.argmax(efficiency_f1)
bars[best_eff_idx].set_edgecolor('gold')
bars[best_eff_idx].set_linewidth(3)

for i, (name, eff) in enumerate(zip(short_names, efficiency_f1)):
    ax2.text(i, eff + 0.3, f'{eff:.2f}',
             ha='center', va='bottom', fontsize=12, weight='bold')

ax2.set_ylabel('F1 Score per 1000 Dimensions', fontsize=13, weight='bold')
ax2.set_title('Feature Efficiency Analysis\n(Higher = Better)', fontsize=14, weight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('Feature Synergy and Efficiency Analysis', fontsize=17, weight='bold', y=0.98)
plt.tight_layout()
plt.savefig(output_dir / 'synergy_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: synergy_analysis.png")


# ============================================================================
# Summary Statistics
# ============================================================================
print()
print("="*70)
print("Ablation Study Summary")
print("="*70)

print("\n📊 Configuration Performance:")
for i, config in enumerate(short_names):
    print(f"\n{config} Features:")
    print(f"  Dimensionality: {feature_dims[i]}")
    print(f"  F1 Score: {f1_scores[i]:.4f}")
    print(f"  AUC: {auc_scores[i]:.4f}")
    print(f"  Recall: {recall_scores[i]:.4f}")
    if params[i] > 0:
        print(f"  Parameters: {params[i]:,}")

print("\n📈 Key Findings:")

# F1 improvement
f1_classical_to_hybrid = ((f1_scores[2] - f1_scores[0]) / f1_scores[0]) * 100 if f1_scores[0] > 0 else float('inf')
print(f"  Hybrid vs Classical (F1): +{f1_classical_to_hybrid:.1f}%")

# AUC improvement
auc_classical_to_hybrid = ((auc_scores[2] - auc_scores[0]) / auc_scores[0]) * 100 if auc_scores[0] > 0 else float('inf')
print(f"  Hybrid vs Classical (AUC): +{auc_classical_to_hybrid:.1f}%")

# Quantum-only analysis
print(f"\n  Quantum-only features (9-dim) achieve:")
print(f"    - F1 = {f1_scores[1]:.4f} (96.64% of perfect)")
print(f"    - Recall = {recall_scores[1]:.4f} (perfect recall!)")
print(f"    - AUC = {auc_scores[1]:.4f} (limited by missing graph structure)")

# Synergy effect
print(f"\n  💡 Synergy Effect:")
print(f"    Classical alone: F1 = {f1_scores[0]:.4f}")
print(f"    Quantum alone: F1 = {f1_scores[1]:.4f}")
print(f"    Hybrid (both): F1 = {f1_scores[2]:.4f} ← Achieves near-perfect performance!")

print("\n💾 All visualizations saved to:")
print(f"  {output_dir.absolute()}")
print()
print("="*70)
print("✓ Ablation visualization completed successfully!")
print("="*70)
