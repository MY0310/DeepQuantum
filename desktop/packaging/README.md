# Packaging Notes

桌面端已改为优先使用 `desktop/` 内部资源，可直接单独打包，不再依赖运行时访问 `ui/` 或 `Deepquantum/` 原目录。

## 当前打包方式

推荐使用 `PyInstaller onedir`：

- 主程序：`dist/QGADDesktop/QGADDesktop.exe`
- 资源目录：已随打包带入 `desktop/assets`、`desktop/resources`、`desktop/storage`

## 一键构建

在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File desktop\packaging\build_desktop.ps1
```

如果普通权限下失败，改用管理员权限终端执行一次。

## 产物位置

- `dist_release/QGADDesktop/QGADDesktop.exe`

## 交付建议

提交时建议压缩整个 `dist_release/QGADDesktop/` 目录，而不是只拿出单个 exe。
