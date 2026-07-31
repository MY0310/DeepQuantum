# Q-GAD Desktop

原生 `PyQt5` 桌面演示端，面向离线答辩与后续 `exe` 打包。

## 运行

```powershell
D:\Tools\Miniconda3\envs\qgad\python.exe desktop\main.py
```

或直接执行：

```powershell
powershell -ExecutionPolicy Bypass -File desktop\run_desktop.ps1
```

```bat
desktop\run_desktop.bat
```

## 特性

- 直接读取 `ui/storage/monitor_bundle.v2.json`
- 三栏式风险研判工作台
- 本地保存处置状态与备注
- 导出告警列表、处置记录、节点摘要
- 可选时序分析弹窗，仅展示比值型指标
