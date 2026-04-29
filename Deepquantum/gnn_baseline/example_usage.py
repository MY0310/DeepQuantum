"""
Example: Quick Comparison Workflow

This script demonstrates the complete workflow for comparing
quantum (GBS) and classical (GNN) models on Elliptic++ dataset.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and print status."""
    print(f"\n{'='*80}")
    print(f"STEP: {description}")
    print(f"{'='*80}")
    print(f"Command: {cmd}\n")

    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[Error] command failed: {cmd}")
        return False
    print(f"[Done] {description}")
    return True


def main():
    """Run complete comparison workflow."""

    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║        Quantum vs Classical GNN: Comparison Workflow                ║
║                                                                      ║
║  This script will:                                                   ║
║  1. Train quantum GBS model (if not already done)                   ║
║  2. Train classical GNN baselines                                    ║
║  3. Generate comparison analysis                                     ║
║  4. Produce report                                                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Configuration
    QUANTUM_SCRIPT = "run_elliptic_fast.py"
    GNN_SCRIPT = "gnn_baseline/run_gnn_baseline.py"
    ANALYSIS_SCRIPT = "gnn_baseline/utils/comparison_analysis.py"

    # Step 1: Run quantum model (optional, uncomment if needed)
    print("\n" + "="*80)
    print("STEP 1: Quantum Model (Optional)")
    print("="*80)
    print("Skipping quantum model training (assuming already done)")
    print("To run: python run_elliptic_fast.py")

    # Step 2: Run GNN baselines (fast mode for quick demo)
    print("\n" + "="*80)
    print("STEP 2: Train Classical GNN Baselines")
    print("="*80)

    # Train GCN (fastest)
    success = run_command(
        f"python {GNN_SCRIPT} --model gcn --epochs 5 --fast",
        "Training GCN Baseline (Fast Mode)"
    )

    if not success:
        print("[Error] GNN training failed. Exiting.")
        sys.exit(1)

    # Train GAT for comparison
    success = run_command(
        f"python {GNN_SCRIPT} --model gat --epochs 5 --fast",
        "Training GAT Baseline (Fast Mode)"
    )

    if not success:
        print("[Warn] GAT training failed, but continuing with GCN results.")

    # Step 3: Generate comparison analysis
    print("\n" + "="*80)
    print("STEP 3: Generate Comparison Analysis")
    print("="*80)

    success = run_command(
        f"python {ANALYSIS_SCRIPT} "
        f"--quantum-dir experiment_summary.json "
        f"--classical-dir gnn_baseline/outputs "
        f"--output-dir gnn_baseline/analysis",
        "Quantum vs Classical Comparison Analysis"
    )

    if not success:
        print("[Error] Analysis failed. Check results paths.")
        sys.exit(1)

    # Final summary
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║                     WORKFLOW COMPLETE                               ║
║                                                                      ║
║  Results saved to:                                                   ║
║    - gnn_baseline/outputs/          (GNN training results)          ║
║    - gnn_baseline/analysis/         (Comparison analysis)           ║
║                                                                      ║
║  Key files to check:                                                ║
║    - gnn_baseline/analysis/metrics_comparison.png                  ║
║    - gnn_baseline/analysis/comparison_report.txt                   ║
║    - gnn_baseline/analysis/comparison_report.md                    ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    print("\nNext steps:")
    print("  1. Review the comparison plots:")
    print("     - gnn_baseline/analysis/metrics_comparison.png")
    print("  2. Read the detailed report:")
    print("     - gnn_baseline/analysis/comparison_report.txt")
    print("  3. For full evaluation (not fast mode), re-run with:")
    print("     python gnn_baseline/run_gnn_baseline.py --model all --epochs 20")


if __name__ == "__main__":
    main()
