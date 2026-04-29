"""
消融实验主入口 (Ablation Study Main Entry)

快捷运行消融实验训练或可视化

Usage:
    python run.py --train           # 训练所有模型
    python run.py --visualize       # 生成论文数据可视化
    python run.py --quick           # 快速验证（2 epochs，子集）

Author: Q-GAD Research Team
Date: 2026-04-15
"""

import sys
from pathlib import Path
import argparse
import subprocess
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="Q-GAD消融实验 (Ablation Study)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py --train              # 训练所有模型（10 epochs）
  python run.py --train --epochs 5   # 训练所有模型（5 epochs）
  python run.py --train --model quantum  # 只训练Quantum-only
  python run.py --visualize          # 生成论文数据可视化
  python run.py --quick              # 快速验证（2 epochs，50%数据）
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--train', action='store_true',
                           help='训练消融实验模型')
    mode_group.add_argument('--visualize', action='store_true',
                           help='生成论文数据可视化图表')

    # Training options
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'classical', 'quantum', 'hybrid'],
                       help='要训练的模型（--train时使用）')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数（默认: 10）')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='学习率（默认: 0.001）')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='批次大小（默认: 32）')
    parser.add_argument('--num-workers', type=int, default=2,
                       help='DataLoader workers（默认: 2，WinError 5 自动回退 0）')
    parser.add_argument('--subset', type=float, default=1.0,
                       help='使用训练集的比例，0.1=10%%（默认: 1.0）')
    parser.add_argument('--max-train-samples', type=int, default=None,
                       help='训练集最大样本数（快速验证/控时）')
    parser.add_argument('--max-val-samples', type=int, default=None,
                       help='验证集最大样本数（快速验证用）')
    parser.add_argument('--max-test-samples', type=int, default=None,
                       help='测试集最大样本数（快速验证用）')
    parser.add_argument('--device', type=str, default=None,
                       help='设备（默认: auto）')
    parser.add_argument('--n-shots', type=int, default=15,
                       help='DeepQuantum每样本采样次数（默认: 15）')
    parser.add_argument('--decision-threshold', type=float, default=0.5,
                       help='固定判定阈值（未开启阈值寻优时生效）')
    parser.add_argument('--optimize-threshold', action='store_true',
                       help='在验证集自动搜索最优F1阈值')
    parser.add_argument('--balance-sampler', action='store_true',
                       help='训练时使用类别均衡采样器')
    parser.add_argument('--no-class-weights', action='store_true',
                       help='关闭类别权重损失')
    parser.add_argument('--patience', type=int, default=3,
                       help='早停容忍轮数（默认: 3）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认: 42）')
    parser.add_argument('--cache-quantum-features', action='store_true',
                       help='将一次性量子特征缓存落盘到 ablation_study/cache/')
    parser.add_argument('--show-quantum-logs', action='store_true',
                       help='显示DeepQuantum详细日志')
    parser.add_argument('--parallel-models', action='store_true',
                       help='当 --model all 时并行启动 classical/quantum/hybrid 三个子进程')
    parser.add_argument('--parallel-jobs', type=int, default=2,
                       help='并行子进程上限（默认: 2）')

    # Quick mode
    parser.add_argument('--quick', action='store_true',
                       help='快速验证模式（等价于 --epochs 2 --subset 0.5）')

    args = parser.parse_args()

    # Apply quick mode
    if args.quick:
        args.epochs = 3
        args.subset = min(args.subset, 0.4)
        args.max_train_samples = args.max_train_samples or 5000
        args.max_val_samples = args.max_val_samples or 2000
        args.max_test_samples = args.max_test_samples or 2000
        args.optimize_threshold = True
        print("[Fast] quick validation mode: 3 epochs, 40%% train subset, limited val/test, threshold tuning\n")

    # Execute
    if args.visualize:
        print("="*70)
        print("生成论文数据可视化图表")
        print("="*70)

        # Import and run visualization
        result = subprocess.run(
            [sys.executable, "-m", "visualization.experiments_dashboard"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=False,
        )

        if result.returncode == 0:
            print("\n[Done] visualization finished. Figures are in outputs/visualizations/experiments/")
        else:
            print("\n[Error] visualization failed.")

    elif args.train:
        if args.model == 'all' and args.parallel_models:
            models = ['classical', 'quantum', 'hybrid']
            max_jobs = max(1, int(args.parallel_jobs))
            print("="*70)
            print("并行训练消融实验模型")
            print("="*70)
            print(f"并行模型: {models}")
            print(f"并发上限: {max_jobs}")
            print("="*70)

            script_path = Path(__file__).resolve()
            project_root = script_path.parent.parent.parent

            def _build_cmd(model_name: str):
                cmd = [
                    sys.executable, str(script_path),
                    '--train',
                    '--model', model_name,
                    '--epochs', str(args.epochs),
                    '--lr', str(args.lr),
                    '--batch-size', str(args.batch_size),
                    '--num-workers', str(args.num_workers),
                    '--subset', str(args.subset),
                    '--n-shots', str(args.n_shots),
                    '--decision-threshold', str(args.decision_threshold),
                    '--patience', str(args.patience),
                    '--seed', str(args.seed),
                ]
                if args.max_train_samples is not None:
                    cmd.extend(['--max-train-samples', str(args.max_train_samples)])
                if args.max_val_samples is not None:
                    cmd.extend(['--max-val-samples', str(args.max_val_samples)])
                if args.max_test_samples is not None:
                    cmd.extend(['--max-test-samples', str(args.max_test_samples)])
                if args.optimize_threshold:
                    cmd.append('--optimize-threshold')
                if args.balance_sampler:
                    cmd.append('--balance-sampler')
                if args.no_class_weights:
                    cmd.append('--no-class-weights')
                if args.cache_quantum_features:
                    cmd.append('--cache-quantum-features')
                if args.device:
                    cmd.extend(['--device', args.device])
                if args.show_quantum_logs:
                    cmd.append('--show-quantum-logs')
                return cmd

            pending = models[:]
            running = []
            failed = []

            while pending or running:
                while pending and len(running) < max_jobs:
                    model_name = pending.pop(0)
                    cmd = _build_cmd(model_name)
                    print(f"[Spawn] {model_name}: {' '.join(cmd)}")
                    proc = subprocess.Popen(cmd, cwd=project_root)
                    running.append((model_name, proc, time.time()))

                time.sleep(1.0)
                still_running = []
                for model_name, proc, t0 in running:
                    ret = proc.poll()
                    if ret is None:
                        still_running.append((model_name, proc, t0))
                        continue
                    elapsed = time.time() - t0
                    if ret == 0:
                        print(f"[Done] {model_name} finished in {elapsed:.1f}s")
                    else:
                        print(f"[Fail] {model_name} exited with code {ret} after {elapsed:.1f}s")
                        failed.append(model_name)
                running = still_running

            if failed:
                print(f"[Error] Parallel training failed: {failed}")
                sys.exit(1)

            print("[Done] Parallel ablation training completed.")
            return

        print("="*70)
        print("训练消融实验模型")
        print("="*70)
        print(f"模型: {args.model}")
        print(f"训练轮数: {args.epochs}")
        print(f"学习率: {args.lr}")
        if args.subset < 1.0:
            print(f"训练子集: {args.subset*100:.1f}%%")
        print(f"批次大小: {args.batch_size}")
        print(f"num_workers: {args.num_workers}")
        print(f"n_shots: {args.n_shots}")
        print(f"decision_threshold: {args.decision_threshold}")
        print(f"optimize_threshold: {args.optimize_threshold}")
        print(f"balance_sampler: {args.balance_sampler}")
        print(f"class_weights: {not args.no_class_weights}")
        print(f"patience: {args.patience}")
        print(f"seed: {args.seed}")
        if args.max_val_samples is not None:
            print(f"max_val_samples: {args.max_val_samples}")
        if args.max_test_samples is not None:
            print(f"max_test_samples: {args.max_test_samples}")
        if args.max_train_samples is not None:
            print(f"max_train_samples: {args.max_train_samples}")
        print("="*70)
        print()

        # Import and run training
        from train_ablation_complete import main as train_main

        # Override sys.argv
        sys.argv = [
            'train_ablation_complete.py',
            '--model', args.model,
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--batch-size', str(args.batch_size),
            '--num-workers', str(args.num_workers),
            '--subset', str(args.subset),
            '--n-shots', str(args.n_shots),
            '--decision-threshold', str(args.decision_threshold),
            '--patience', str(args.patience),
            '--seed', str(args.seed),
        ]
        if args.max_train_samples is not None:
            sys.argv.extend(['--max-train-samples', str(args.max_train_samples)])
        if args.max_val_samples is not None:
            sys.argv.extend(['--max-val-samples', str(args.max_val_samples)])
        if args.max_test_samples is not None:
            sys.argv.extend(['--max-test-samples', str(args.max_test_samples)])
        if args.optimize_threshold:
            sys.argv.append('--optimize-threshold')
        if args.balance_sampler:
            sys.argv.append('--balance-sampler')
        if args.no_class_weights:
            sys.argv.append('--no-class-weights')
        if args.cache_quantum_features:
            sys.argv.append('--cache-quantum-features')

        if args.device:
            sys.argv.extend(['--device', args.device])
        if args.show_quantum_logs:
            sys.argv.append('--show-quantum-logs')

        train_main()


if __name__ == "__main__":
    main()
