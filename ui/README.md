# Q-GAD Demo UI

离线优先的时序拓扑风险监控台（Streamlit）。

## Quick Start

```powershell
# 构建监控数据包
conda run -n qgad python ui\scripts\build_monitor_bundle.py

# 校验数据包
conda run -n qgad python ui\scripts\validate_monitor_bundle.py

# 可选自检
conda run -n qgad python ui\scripts\smoke_check.py

# Launch UI
conda run -n qgad python -m streamlit run ui\app.py
```

## Structure

- `app.py`: 监控台单页入口（时间回放 + 拓扑监控）
- `config/`: 路径、常量、主题
- `data/`: 类型定义与数据包加载
- `services/`: 时序快照、风险子图、节点详情、可选实时复核
- `components/`: 图表与业务面板
- `scripts/`: 构建、校验、自检
- `storage/`: 生成的 `monitor_bundle.v2.json`
- `assets/`: 本地样式
