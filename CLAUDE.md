# maskperson

视频人体脱敏工具 — YOLOv8-Seg 跟踪 + 像素级马赛克。

## 项目结构

```
src/maskperson/       # 源码
tests/                # 单元测试
configs/             # 配置文件
scripts/run.sh       # 快速运行
models/              # YOLO 权重
input/                # 输入视频
output/               # 输出视频
temp/                 # 临时文件
```

## 运行

使用 `.venv/bin/python` 而非 `uv run`（uv run 在多平台解析 torch 依赖时有冲突）。

```bash
.venv/bin/python -m maskperson -i input/input.mp4 -o output/anonymized.mp4
./scripts/run.sh input.mp4 output.mp4
```

## 依赖问题

torch 默认从 PyPI 拉取（兼容 GPU/CPU 的 fat wheel），无 NVIDIA 机器运行时自动降级 CPU。如需更小体积的纯 CPU wheel（避开 NVIDIA 驱动版本要求）：
```bash
uv sync --no-extra default --extra cpu
```

## 设备选择

启动时调用 `maskperson.device.resolve_device(cfg.device)` 决定推理设备，支持 `auto` / `cpu` / `cuda` / `cuda:N`。CLI 用 `--device` 显式覆盖。NVIDIA 机器自动用 GPU；无 GPU 机器自动降级 CPU。

## 测试

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m mypy src/
```
