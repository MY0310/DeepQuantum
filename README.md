# Q-GAD Project

最后更新：2026-04-29

本仓库的主工程位于 `Deepquantum/` 目录。

## 命名说明

- 外层 `qgad-project/`：仓库根（repo root）
- 内层 `Deepquantum/`：主工程根（app root）

为减少歧义，文档统一使用 `repo root` 与 `app root` 两个术语。

## 目录入口

```text
.
├── Deepquantum/          # 主工程（代码、实验、基线、可视化）
├── article/              # 论文与材料
├── dq.ps1                # repo root 快捷转发到 app root
└── README.md             # 本文件（仓库级索引）
```

## 主工程文档

- 工程总览：`Deepquantum/README.md`
- 实验总览：`Deepquantum/experiments/README.md`
- 实验结果：`Deepquantum/experiments/RESULTS.md`
- GNN 基线：`Deepquantum/gnn_baseline/README.md`

## 快速进入主工程

```bash
cd Deepquantum
```

## 根目录快捷执行（可选）

在仓库根可直接用 `dq.ps1` 转发命令到主工程根，无需手动 `cd`：

```powershell
.\dq.ps1 python run_elliptic.py --help
```

## 清理约定

- 已清理：`__pycache__/`、`.mplconfig/`（本地运行缓存）
- 保留：`experiments/*/cache/*.pt`（实验复用缓存，可显著减少重复运行时间）
